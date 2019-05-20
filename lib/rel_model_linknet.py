"""
LinkNet reimplemented by Jiayan (jiayanyang97@gmail.com)

Adapted from https://github.com/rowanz/neural-motifs/blob/master/lib/rel_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM

from lib.fpn.box_utils import bbox_overlaps, center_size, nms_overlaps
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector_2 import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import to_onehot, arange, diagonal_inds, Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math


MODES = ('sgdet', 'sgcls', 'predcls')


class GlobalContextEncoding(nn.Module):
    """
    Module for global context encoding
    """
    def __init__(self, num_classes, ctx_dim=512):
        super(GlobalContextEncoding, self).__init__()

        self.glb_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flattener(),
        )

        self.multi_score_fc = nn.Linear(ctx_dim, num_classes)

    def forward(self, features):
        """
        Forward pass for global context encoding
        :param features: [batch_size, ctx_dim, IM_SIZE/4, IM_SIZE/4] fmap features
        :return: c: [batch_size, ctx_dim] context feature
                 M: [batch_size, num_classes] softmax of multi-label distribution
        """
        c = self.glb_avg_pool(features)
        M = F.softmax(self.multi_score_fc(c), dim=1)
        return c, M


class RelationalEmbedding(nn.Module):
    """
    Module for relational embedding
    """
    def __init__(self, input_dim, output_dim, r=2):
        super(RelationalEmbedding, self).__init__()

        self.W = nn.Linear(input_dim, int(input_dim/r))
        self.U = nn.Linear(input_dim, int(input_dim/r))
        self.H = nn.Linear(input_dim, int(input_dim/r))

        self.fc0 = nn.Linear(int(input_dim/r), input_dim)
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, O0):
        """
        Forward pass for relational embedding
        :param O0: [N, input_dim] object features
        :return: O1: [N, input_dim] encoded features
                 O2: [N, output_dim] decoded features
        """
        R1 = F.softmax(torch.matmul(self.W(O0), torch.t(self.U(O0))), 1)
        O1 = O0 + self.fc0(torch.matmul(R1, self.H(O0)))
        O2 = self.fc1(O1)
        return O1, O2


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, rel_classes, mode='sgcls',
                 embed_dim=200, hidden_dim=256, obj_dim=4096, pooling_dim=4096, ctx_dim=512):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.pooling_dim = pooling_dim
        self.ctx_dim = ctx_dim

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)  # K0
        self.obj_embed.weight.data = embed_vecs.clone()
        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)  # K1
        self.obj_embed2.weight.data = embed_vecs.clone()

        # Object-Relational Embedding
        self.RE1 = RelationalEmbedding(input_dim=self.obj_dim+self.embed_dim+self.ctx_dim, output_dim=self.hidden_dim)
        self.RE2 = RelationalEmbedding(input_dim=self.hidden_dim, output_dim=self.num_classes)

        # Edge-Relational Embedding
        self.RE3 = RelationalEmbedding(input_dim=self.embed_dim+self.hidden_dim, output_dim=self.hidden_dim)
        self.RE4 = RelationalEmbedding(input_dim=self.hidden_dim, output_dim=self.pooling_dim*2)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def get_max_preds(self, obj_dists, obj_labels, boxes_for_nms):
        """
        Get max non-background prediction
        :param obj_dists: [num_obj, num_classes] new probability distribution: O4
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes_for_nms: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_preds: [num_obj] argmax of that distribution: O4'
        """
        if self.training:
            # Whenever labels are 0 set to be max prediction
            obj_preds = obj_labels
            nonzero_pred = obj_dists[:, 1:].max(1)[1] + 1
            is_bg = (obj_preds.data == 0).nonzero()
            if is_bg.dim() > 0:
                obj_preds[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
        else:
            # Greedily take the max here amongst non-bgs
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        # when sgdet is testing, do NMS as a post-processing step
        if boxes_for_nms is not None and not self.training:
            nms_thresh = 0.3
            is_overlap = nms_overlaps(boxes_for_nms.data).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= nms_thresh

            obj_preds = obj_preds[0].data.new(len(obj_preds)).fill_(0)
            out_dists_sampled = F.softmax(obj_dists, dim=1).data.cpu().numpy()
            out_dists_sampled[:, 0] = 0

            for i in range(obj_preds.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                obj_preds[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample
            obj_preds = Variable(obj_preds)

        return obj_preds

    def obj_ctx(self, obj_feats, obj_labels=None, boxes_for_nms=None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, obj_dim + embed_dim + ctx_dim]: O0
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes_for_nms: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, num_classes] new probability distribution: O4
                 obj_preds: [num_obj] argmax of that distribution: O4'
                 obj_ctx: [num_obj, hidden_dim] for later edge contex: O3
        """
        O1, O2 = self.RE1(obj_feats)
        obj_ctx, obj_dists = self.RE2(O2)

        if self.mode != 'predcls':
            obj_preds = self.get_max_preds(obj_dists, obj_labels, boxes_for_nms)
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data, self.num_classes))

        return obj_dists, obj_preds, obj_ctx

    def edge_ctx(self, obj_ctx, obj_preds):
        """
        Edge context and edge representation.
        :param obj_ctx: [num_obj, hidden_dim]: O3
        :param obj_preds: [num_obj] argmax of new distribution: O4'
        :return: edge_ctx: [num_obj, hidden_dim]
                 edge_rep: [num_obj, pooling_dim * 2] for later subject and object representations: E1
        """
        obj_embed2 = self.obj_embed2(obj_preds)
        inp_feats = torch.cat((obj_embed2, obj_ctx), 1)  # E0

        tmp1, tmp2 = self.RE3(inp_feats)
        edge_ctx, edge_rep = self.RE4(tmp2)

        return edge_ctx, edge_rep

    def forward(self, obj_fmap, obj_dists, context_features, obj_labels=None, boxes_for_nms=None):
        """
        Forward pass through the object and edge context
        :param obj_fmap: [num_obj, obj_dim]: ROI-aligned features: fi^ROI
        :param obj_dists: [num_obj, num_classes] object label distribution from detector: li
        :param context_features: [num_obj, ctx_dim] image-level context features: stack of c
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes_for_nms: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists2: [num_obj, num_classes] new distribution after contex: O4
                 obj_preds: [num_obj] argmax of that distribution: O4'
                 edge_rep: [num_obj, pooling_dim * 2] for later subject and object representations: E1
        """

        obj_embed = F.softmax(obj_dists, dim=1) @ self.obj_embed.weight
        obj_feats = torch.cat((obj_fmap, obj_embed, context_features), 1)  # O0

        obj_dists2, obj_preds, obj_ctx = self.obj_ctx(obj_feats, obj_labels, boxes_for_nms)
        edge_ctx, edge_rep = self.edge_ctx(obj_ctx, obj_preds)

        return obj_dists2, obj_preds, edge_rep


class RelModelLinknet(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=4096,
                 nl_obj=1, nl_edge=2, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModelLinknet, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.ctx_dim = 1024 if use_resnet else 512
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim, pooling_dim=self.pooling_dim, ctx_dim=self.ctx_dim)

        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        # Global Context Encoding
        self.GCE = GlobalContextEncoding(num_classes=self.num_classes, ctx_dim=self.ctx_dim)

        ###################################

        # K2
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        # fc4
        self.rel_compress = nn.Linear(self.pooling_dim+128, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        if self.use_bias:
            self.freq_bias = FrequencyBias()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def geo_layout_enc(self, box_priors, rel_inds):
        """
        Geometric Layout Encoding
        :param box_priors: [num_rois, 4] of (xmin, ymin, xmax, ymax)
        :param rel_inds: [num_rels, 3] of (img ind, box0 ind, box1 ind)
        :return: bos: [num_rois*(num_rois-1), 4] encoded relative geometric layout: bo|s
        """
        cxcywh = center_size(box_priors.data)  # convert to (cx, cy, w, h)
        box_s = cxcywh[rel_inds[:, 1]]
        box_o = cxcywh[rel_inds[:, 2]]

        # relative location
        rlt_loc_x = torch.div((box_o[:, 0] - box_s[:, 0]), box_s[:, 2]).view(-1, 1)
        rlt_loc_y = torch.div((box_o[:, 1] - box_s[:, 1]), box_s[:, 3]).view(-1, 1)

        # scale information
        scl_info_w = torch.log(torch.div(box_o[:, 2], box_s[:, 2])).view(-1, 1)
        scl_info_h = torch.log(torch.div(box_o[:, 3], box_s[:, 3])).view(-1, 1)

        bos = torch.cat((rlt_loc_x, rlt_loc_y, scl_info_w, scl_info_h), 1)
        return bos

    def glb_context_enc(self, features, im_inds, gt_classes, image_offset):
        """
        Global Context Encoding
        :param features: [batch_size, ctx_dim, IM_SIZE/4, IM_SIZE/4] fmap features
        :param im_ind: [num_rois] image index
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :return: context_features: [num_rois, ctx_dim] stacked context_feature c according to im_ind
                 gce_obj_dists: [batch_size, num_classes] softmax of predicted multi-label distribution: M
                 gce_obj_labels: [batch_size, num_classes] ground truth multi-labels
        """
        context_feature, gce_obj_dists = self.GCE(features)
        context_features = context_feature[im_inds]

        gce_obj_labels = torch.zeros_like(gce_obj_dists)
        gce_obj_labels[gt_classes[:, 0] - image_offset, gt_classes[:, 1]] = 1

        return context_features, gce_obj_dists, gce_obj_labels

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for relationship
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # c M
        context_features, result.gce_obj_dists, result.gce_obj_labels = self.glb_context_enc(result.fmap.detach(),
                                                                                             im_inds.data,
                                                                                             gt_classes.data,
                                                                                             image_offset)

        # Prevent gradients from flowing back into score_fc from elsewhere
        result.rm_obj_dists, result.obj_preds, edge_rep = self.context(result.obj_fmap,
                                                                       result.rm_obj_dists.detach(),
                                                                       context_features.detach(),
                                                                       result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
                                                                       result.boxes_all)

        # Split into subject and object representations
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.pooling_dim)  # E1
        subj_rep = edge_rep[:, 0]  # E1_s
        obj_rep = edge_rep[:, 1]  # E1_o

        prod_rep = subj_rep[rel_inds[:, 1]] * obj_rep[rel_inds[:, 2]]  # G0

        if self.use_vision:
            vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])  # F
            if self.limit_vision:
                # exact value TBD
                prod_rep = torch.cat((prod_rep[:,:2048] * vr[:,:2048], prod_rep[:,2048:]), 1)
            else:
                prod_rep = prod_rep * vr

        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)

        bos = self.geo_layout_enc(boxes, rel_inds)  # bo|s
        pos_embed = self.pos_embed(Variable(bos))

        result.rel_dists = self.rel_compress(torch.cat((prod_rep, pos_embed), 1))  # G2

        if self.use_bias:
            result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))

        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
