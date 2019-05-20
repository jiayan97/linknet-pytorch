# Just some tests so you can be assured that sg_eval.py works the same as the (original) stanford evaluation

import numpy as np
from six.moves import xrange
from dataloaders.visual_genome import VG
from lib.evaluation.sg_eval import evaluate_from_dict
from tqdm import trange
from lib.fpn.box_utils import center_size, point_form
def eval_relation_recall(sg_entry,
                         roidb_entry,
                         result_dict,
                         mode,
                         iou_thresh):

    # gt
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_relations[:,2],
                                             gt_relations[:,:2],
                                             gt_classes,
                                             gt_boxes,
                                             gt_predicate_scores,
                                             gt_class_scores)

    # pred
    box_preds = sg_entry['boxes']
    num_boxes = box_preds.shape[0]
    predicate_preds = sg_entry['relations']
    class_preds = sg_entry['scores']
    predicate_preds = predicate_preds.reshape(num_boxes, num_boxes, -1)

    # no bg
    predicate_preds = predicate_preds[:, :, 1:]
    predicates = np.argmax(predicate_preds, 2).ravel() + 1
    predicate_scores = predicate_preds.max(axis=2).ravel()
    relations = []
    keep = []
    for i in xrange(num_boxes):
        for j in xrange(num_boxes):
            if i != j:
                keep.append(num_boxes*i + j)
                relations.append([i, j])
    # take out self relations
    predicates = predicates[keep]
    predicate_scores = predicate_scores[keep]

    relations = np.array(relations)
    assert(relations.shape[0] == num_boxes * (num_boxes - 1))
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]

    if mode =='predcls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert(num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode =='sgcls':
        assert(num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = gt_boxes
    elif mode =='sgdet':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = []
        for i, c in enumerate(classes):
            boxes.append(box_preds[i]) # no bbox regression, c*4:(c+1)*4])
        boxes = np.vstack(boxes)
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)


    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    for k in result_dict[mode + '_recall']:
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds,:],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds,:],
                                  iou_thresh)
        result_dict[mode + '_recall'][k].append(recall)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores):

    # format predictions into triplets
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in xrange(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i,:2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score =  class_scores[sub_i]
        score *= class_scores[obj_i]
        score *= predicate_scores[i]
        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores


def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh):

    # compute the R@K metric for a set of predicted triplets

    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep,:]
        sub_iou = iou(gt_box[:4], boxes[:,:4])
        obj_iou = iou(gt_box[4:], boxes[:,4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt) / float(num_gt)


def iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:,0])
    iymin = np.maximum(gt_box[1], pred_boxes[:,1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:,2])
    iymax = np.minimum(gt_box[3], pred_boxes[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
            (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
            (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

train, val, test = VG.splits()

result_dict_mine = {'sgdet_recall': {20: [], 50: [], 100: []}}
result_dict_theirs = {'sgdet_recall': {20: [], 50: [], 100: []}}

for img_i in trange(len(val)):
    gt_entry = {
        'gt_classes': val.gt_classes[img_i].copy(),
        'gt_relations': val.relationships[img_i].copy(),
        'gt_boxes': val.gt_boxes[img_i].copy(),
    }

    # Use shuffled GT boxes
    gt_indices = np.arange(gt_entry['gt_boxes'].shape[0]) #np.random.choice(gt_entry['gt_boxes'].shape[0], 20)
    pred_boxes = gt_entry['gt_boxes'][gt_indices]

    # Jitter the boxes a bit
    pred_boxes = center_size(pred_boxes)
    pred_boxes[:,:2] += np.random.rand(pred_boxes.shape[0], 2)*128
    pred_boxes[:,2:] *= (1+np.random.randn(pred_boxes.shape[0], 2).clip(-0.1, 0.1))
    pred_boxes = point_form(pred_boxes)

    obj_scores = np.random.rand(pred_boxes.shape[0])

    rels_to_use = np.column_stack(np.where(1 - np.diag(np.ones(pred_boxes.shape[0], dtype=np.int32))))
    rel_scores = np.random.rand(min(100, rels_to_use.shape[0]), 51)
    rel_scores = rel_scores / rel_scores.sum(1, keepdims=True)
    pred_rel_inds = rels_to_use[np.random.choice(rels_to_use.shape[0], rel_scores.shape[0],
                                                               replace=False)]

    # We must sort by P(o, o, r)
    rel_order = np.argsort(-rel_scores[:,1:].max(1) * obj_scores[pred_rel_inds[:,0]] * obj_scores[pred_rel_inds[:,1]])

    pred_entry = {
        'pred_boxes': pred_boxes,
        'pred_classes': gt_entry['gt_classes'][gt_indices], #1+np.random.choice(150, pred_boxes.shape[0], replace=True),
        'obj_scores': obj_scores,
        'pred_rel_inds': pred_rel_inds[rel_order],
        'rel_scores': rel_scores[rel_order],
    }

    # def check_whether_they_are_the_same(gt_entry, pred_entry):
    evaluate_from_dict(gt_entry, pred_entry, 'sgdet', result_dict_mine, multiple_preds=False,
                       viz_dict=None)

    #########################
    predicate_scores_theirs = np.zeros((pred_boxes.shape[0], pred_boxes.shape[0], 51), dtype=np.float64)
    for (o1, o2), s in zip(pred_entry['pred_rel_inds'], pred_entry['rel_scores']):
        predicate_scores_theirs[o1, o2] = s

    obj_scores_theirs = np.zeros((obj_scores.shape[0], 151), dtype=np.float64)
    obj_scores_theirs[np.arange(obj_scores.shape[0]), pred_entry['pred_classes']] = obj_scores

    sg_entry_orig_format = {
        'boxes': pred_entry['pred_boxes'],
        # 'gt_classes': gt_entry['gt_classes'],
        # 'gt_relations': gt_entry['gt_relations'],
        'relations': predicate_scores_theirs,
        'scores': obj_scores_theirs
    }
    roidb_entry = {
        'max_overlaps': np.concatenate((np.ones(gt_entry['gt_boxes'].shape[0]), np.zeros(pred_entry['pred_boxes'].shape[0])), 0),
        'boxes': np.concatenate((gt_entry['gt_boxes'], pred_entry['pred_boxes']), 0),
        'gt_classes': gt_entry['gt_classes'],
        'gt_relations': gt_entry['gt_relations'],
    }
    eval_relation_recall(sg_entry_orig_format, roidb_entry, result_dict_theirs, 'sgdet', iou_thresh=0.5)

my_results = np.array(result_dict_mine['sgdet_recall'][20])
their_results = np.array(result_dict_theirs['sgdet_recall'][20])

assert np.all(my_results == their_results)