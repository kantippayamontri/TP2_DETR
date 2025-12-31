import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as patches

def iou_visualize():
    # Load JSON results and annotations
    with open('../../results/Thumos14/Thumos14_Draw_split0_0525_1258/detection_test_12_raw.json', 'r') as f:
        results_new = json.load(f)['results']
    with open('../../../GAP/results/Thumos14/Thumos14_Draw_split0_0525_2318/detection_test_40_raw.json', 'r') as f:
        results_base = json.load(f)['results']
    with open('../../results/Thumos14/Thumos14_Draw_split0_baseline_0526_2115/detection_test_25_raw.json', 'r') as f:
        results_base2 = json.load(f)['results']
    with open('../../../GAP/data/Thumos14/Thumos14_annotations.json', 'r') as f:
        annotations = json.load(f)['database']

    # IoU calculation
    def iou(seg1, seg2):
        inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
        return inter / union if union > 0 else 0

    # Extract and filter short GT segments (<= 5%)
    short_threshold = 0.05
    gt_short = {}
    for vid, info in annotations.items():
        if vid.startswith('video_test'):
            segs = [ann['segment'] for ann in info['annotations']]
            duration = info["duration"]
            short_segs = [seg for seg in segs if (seg[1] - seg[0])/duration <= short_threshold]
            if short_segs:
                gt_short[vid] = short_segs

    # Compute average IoU improvement on short segments
    improvements = []
    for vid, segs in gt_short.items():
        k = len(segs)
        if vid in results_new and vid in results_base:
            top_new = sorted(results_new[vid], key=lambda x: x['score'], reverse=True)[:k]
            top_base = sorted(results_base[vid], key=lambda x: x['score'], reverse=True)[:k]
            top_base2 = sorted(results_base2[vid], key=lambda x: x['score'], reverse=True)[:k]
            avg_iou_new = sum(max(iou(p['segment'], g) for p in top_new) for g in segs) / k
            avg_iou_base = sum(max(iou(p['segment'], g) for p in top_base) for g in segs) / k
            avg_iou_base2 = sum(max(iou(p['segment'], g) for p in top_base2) for g in segs) / k
            improvements.append((vid, (avg_iou_new - avg_iou_base2)+(avg_iou_new - avg_iou_base), k, avg_iou_new, avg_iou_base, avg_iou_base2))

    # Sort by IoU improvement and pick top 3
    improvements.sort(key=lambda x: x[1], reverse=True)
    top3 = improvements[:3]

    # Visualize the top-3 videos
    for vid, delta_iou, k, new_iou, base_iou, base2_iou in top3:
        segs = gt_short[vid]
        top_new = sorted(results_new[vid], key=lambda x: x['score'], reverse=True)[:k]
        top_base = sorted(results_base[vid], key=lambda x: x['score'], reverse=True)[:k]
        top_base2 = sorted(results_base2[vid], key=lambda x: x['score'], reverse=True)[:k]
        
        fig, ax = plt.subplots(figsize=(14, 2))
        # Plot GT at y=3
        for seg in segs:
            ax.hlines(2, seg[0], seg[1], linewidth=8, label='Ground Truth' if seg == segs[0] else '', colors='black')
            # 在 seg[0] 和 seg[1] 處各畫一條虛線
            ax.vlines(seg[0], ymin=0, ymax=2, linestyles='dashed', linewidth=1, colors=(0, 0, 0, 0.1))
            ax.vlines(seg[1], ymin=0, ymax=2, linestyles='dashed', linewidth=1, colors=(0, 0, 0, 0.1))

        # # Plot Baseline at y=2
        # for p in top_base:
        #     ax.hlines(1.5, p['segment'][0], p['segment'][1], linewidth=8, label='GAP' if p == top_base[0] else '', colors=(0, 0, 0, 0.7))
        # # Plot Baseline2 at y=1
        # for p in top_base2:
        #     ax.hlines(1, p['segment'][0], p['segment'][1], linewidth=8, label='Baseline' if p == top_base2[0] else '', colors=(0, 0, 0, 0.3))

        # Plot Baseline at y=2
        for p in top_base:
            rect = patches.Rectangle(
                (p['segment'][0], 1.5 - 0.2),
                p['segment'][1] - p['segment'][0],
                0.3,
                linewidth=0,
                facecolor='lightgray',
                hatch='///////////',
                label='GAP' if p == top_base[0] else ''
            )
            ax.add_patch(rect)
        # Plot Baseline2 at y=1
        for p in top_base2:
            rect = patches.Rectangle(
                (p['segment'][0], 1 - 0.2),
                p['segment'][1] - p['segment'][0],
                0.3,
                linewidth=0,
                facecolor='gainsboro',
                hatch='\\\\\\\\\\\\',
                label='Baseline' if p == top_base2[0] else ''
            )
            ax.add_patch(rect)
        # Plot Temporal FPN at y=0
        for p in top_new:
            ax.hlines(0.5, p['segment'][0], p['segment'][1], linewidth=8, label='TP²-DETR' if p == top_new[0] else '', colors=(253/256, 222/256, 134/256))
        
        ax.set_yticks([0.5, 1, 1.5, 2])
        ax.set_yticklabels(['TP²-DETR', 'Baseline', 'GAP', 'Ground Truth'])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'(1) {vid} | k={k}, (AvgIoU: TP²-DETR {new_iou:.2f} vs GAP {base_iou:.2f} vs Baseline {base2_iou:.2f})')
        ax.set_ylim(0, 2.5)
        # ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax.legend(loc=(1.02, 0))
        plt.tight_layout()
        plt.savefig(f'qualitative_iou_{vid}.png')


def recall_visualize():
    # Load JSON files
    with open('../../results/Thumos14/Thumos14_Draw_split0_0525_1258/detection_test_12_raw.json', 'r') as f:
        data12 = json.load(f)
    with open('../../../GAP/results/Thumos14/Thumos14_Draw_split0_0525_2318/detection_test_40_raw.json', 'r') as f:
        data40 = json.load(f)
    with open('../../../GAP/data/Thumos14/Thumos14_annotations.json', 'r') as f:
        annotations = json.load(f)


    # Extract test-set annotations
    gt_test = {
        vid: info['annotations']
        for vid, info in annotations['database'].items()
        if vid.startswith('video_test')
    }

    # IoU calculation
    def iou(seg1, seg2):
        start1, end1 = seg1
        start2, end2 = seg2
        inter = max(0, min(end1, end2) - max(start1, start2))
        union = (end1 - start1) + (end2 - start2) - inter
        return inter / union if union > 0 else 0

    # Recall at IoU threshold
    def recall(preds, gt_segments, iou_thresh=0.5):
        if not gt_segments:
            return 0.0
        detected = sum(
            any(iou(p['segment'], g) >= iou_thresh for p in preds)
            for g in gt_segments
        )
        return detected / len(gt_segments)

    # Compute improvements per video
    improvements = []
    for vid, preds12 in data12['results'].items():
        if vid in data40['results'] and vid in gt_test:
            preds40 = data40['results'][vid]
            gt_segs = [ann['segment'] for ann in gt_test[vid]]
            rec12 = recall(preds12, gt_segs)
            rec40 = recall(preds40, gt_segs)
            improvements.append((vid, rec12 - rec40, rec12, rec40))

    # Sort by improvement
    improvements.sort(key=lambda x: x[1], reverse=True)
    top_videos = improvements[:3]

    # Visualize top 3
    for vid, imp, rec12, rec40 in top_videos:
        preds12 = sorted(data12['results'][vid], key=lambda x: x['score'], reverse=True)[:3]
        preds40 = sorted(data40['results'][vid], key=lambda x: x['score'], reverse=True)[:3]
        gt_segs = [ann['segment'] for ann in gt_test[vid]]

        fig, ax = plt.subplots(figsize=(10, 2))
        # Plot GT
        for seg in gt_segs:
            ax.hlines(2, seg[0], seg[1], linewidth=8)
        # Plot baseline
        for seg in preds40:
            ax.hlines(1, seg['segment'][0], seg['segment'][1], linewidth=6)
        # Plot improved model
        for seg in preds12:
            ax.hlines(0, seg['segment'][0], seg['segment'][1], linewidth=6)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Temporal FPN', 'GAP', 'Ground Truth'])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Video: {vid} | Recall Δ@0.5 IoU: {imp:.2f} (New: {rec12:.2f}, Base: {rec40:.2f})')
        ax.set_ylim(-1, 3)
        plt.tight_layout()
        plt.savefig(f'qualitative_recall_{vid}.png')


def scatter_visualize():
    # Load JSON results and annotations
    with open('../../results/Thumos14/Thumos14_Draw_split0_0525_1258/detection_test_12_raw.json', 'r') as f:
        results_new = json.load(f)['results']
    # with open('../../../GAP/results/Thumos14/Thumos14_Draw_split0_0525_2318/detection_test_40_raw.json', 'r') as f:
    #     results_base = json.load(f)['results']
    with open('../../results/Thumos14/Thumos14_Draw_split0_baseline_0526_2115/detection_test_25_raw.json', 'r') as f:
        results_base = json.load(f)['results']
    with open('../../../GAP/data/Thumos14/Thumos14_annotations.json', 'r') as f:
        annotations = json.load(f)['database']

    # IoU calculation
    def iou(seg1, seg2):
        inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
        return inter / union if union > 0 else 0

    # Collect scores and IoUs
    scores_base, ious_base = [], []
    scores_new, ious_new = [], []

    for vid, info in annotations.items():
        if not vid.startswith('video_test'):
            continue
        gt_segs = [ann['segment'] for ann in info['annotations']]
        for p in results_base.get(vid, []):
            best_iou = max(iou(p['segment'], g) for g in gt_segs) if gt_segs else 0
            scores_base.append(p['score'])
            ious_base.append(best_iou)
        for p in results_new.get(vid, []):
            best_iou = max(iou(p['segment'], g) for g in gt_segs) if gt_segs else 0
            scores_new.append(p['score'])
            ious_new.append(best_iou)

    # Plot with distinct colors
    plt.figure(figsize=(8, 6))
    # plt.scatter(scores_base, ious_base, s=5, alpha=0.4, color='blue', label='Baseline')
    plt.scatter(scores_new, ious_new, s=5, alpha=0.4, color='red', label='TP²-DETR')
    plt.xlabel('Confidence Score')
    plt.ylabel('Best IoU with GT')
    plt.title('Confidence vs. IoU Comparison')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'qualitative_scatter2.png')


def map_visualize(threshold = 0.5):
    
    # Load JSON results and annotations
    with open('../../results/Thumos14/Thumos14_Draw_split0_0525_1258/detection_test_12_raw.json', 'r') as f:
        data_new = json.load(f)['results']
    # with open('../../../GAP/results/Thumos14/Thumos14_Draw_split0_0525_2318/detection_test_40_raw.json', 'r') as f:
    #     data_base = json.load(f)['results']
    with open('../../results/Thumos14/Thumos14_Draw_split0_baseline_0526_2115/detection_test_25_raw.json', 'r') as f:
        data_base = json.load(f)['results']
    with open('../../../GAP/data/Thumos14/Thumos14_annotations.json', 'r') as f:
        ann = json.load(f)['database']

    # Extract test set GT
    gt = {vid: info['annotations'] for vid, info in ann.items() if vid.startswith('video_test')}

    # IoU function
    def iou(seg1, seg2):
        inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
        return inter / union if union > 0 else 0

    # Compute AP for one video at IoU threshold
    def average_precision(preds, gts, iou_thr=threshold):
        if len(gts) == 0:
            return np.nan
        # Sort preds by score descending
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        tp = np.zeros(len(preds_sorted))
        fp = np.zeros(len(preds_sorted))
        matched = np.zeros(len(gts), dtype=bool)
        
        for i, p in enumerate(preds_sorted):
            ious = np.array([iou(p['segment'], g['segment']) for g in gts])
            max_iou_idx = ious.argmax() if len(ious) > 0 else -1
            if len(ious) > 0 and ious[max_iou_idx] >= iou_thr and not matched[max_iou_idx]:
                tp[i] = 1
                matched[max_iou_idx] = True
            else:
                fp[i] = 1
        
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall = cum_tp / len(gts)
        
        # AP: sum over (R_n - R_{n-1}) * P_n
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precision, recall):
            ap += p * (r - prev_recall)
            prev_recall = r
        return ap, precision, recall

    # Compute per-video AP and improvements
    results = []
    for vid in gt.keys():
        ap_new, _, _ = average_precision(data_new.get(vid, []), gt[vid], 0.5)
        ap_base, _, _ = average_precision(data_base.get(vid, []), gt[vid], 0.5)
        if not np.isnan(ap_new) and not np.isnan(ap_base):
            results.append((vid, ap_new - ap_base, ap_new, ap_base))

    # Sort by improvement
    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:10]

    # Bar chart of AP
    videos = [r[0] for r in top3]
    ap_new_vals = [r[2] for r in top3]
    ap_base_vals = [r[3] for r in top3]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(videos))
    width = 0.35
    ax.bar(x - width/2, ap_base_vals, width, label='Baseline')
    ax.bar(x + width/2, ap_new_vals, width, label='TP²-DETR')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=45)
    ax.set_ylabel('Average Precision (AP) @ IoU=0.5')
    ax.set_title('Per-video AP Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'qualitative_mapStat.png')

    # Precision-Recall curves
    for vid, _, _, _ in top3:
        ap_new, prec_new, rec_new = average_precision(data_new[vid], gt[vid], 0.5)
        ap_base, prec_base, rec_base = average_precision(data_base[vid], gt[vid], 0.5)

        # Sort for area plot
        rec_new_sorted, prec_new_sorted = zip(*sorted(zip(rec_new, prec_new)))
        rec_base_sorted, prec_base_sorted = zip(*sorted(zip(rec_base, prec_base)))

        plt.figure(figsize=(6, 4))
        plt.plot(rec_base_sorted, prec_base_sorted, label=f'Baseline mAP@{threshold}={ap_base:.2f}')
        plt.fill_between(rec_base_sorted, prec_base_sorted, alpha=0.2)
        plt.plot(rec_new_sorted, prec_new_sorted, label=f'TP²-DETR mAP@{threshold}={ap_new:.2f}')
        plt.fill_between(rec_new_sorted, prec_new_sorted, alpha=0.2)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall: {vid}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'qualitative_map_{vid}.png')

# -----------------------------------------------------------

def iou_matching_visualize():

    # Load JSON results and annotations
    with open('../../results/Thumos14/Thumos14_Draw_split0_0525_1258/detection_test_12_raw.json', 'r') as f:
        results_new = json.load(f)['results']
    with open('../../../GAP/results/Thumos14/Thumos14_Draw_split0_0525_2318/detection_test_40_raw.json', 'r') as f:
        results_base = json.load(f)['results']
    with open('../../results/Thumos14/Thumos14_Draw_split0_baseline_0526_2115/detection_test_25_raw.json', 'r') as f:
        results_base2 = json.load(f)['results']
    with open('../../../GAP/data/Thumos14/Thumos14_annotations.json', 'r') as f:
        annotations = json.load(f)['database']

    vids = ['video_test_0000464', 'video_test_0000577', 'video_test_0001038']

    lambda_cls = 2
    lambda_l1 = 5
    lambda_iou = 2
    alpha = 0.25
    gamma = 2.0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def iou(seg1, seg2):
        inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        union = (seg1[1] - seg1[0]) + (seg2[1] - seg1[0]) - inter
        return inter / union if union > 0 else 0

    def build_cost_numpy(pred_scores, pred_boxes, tgt_boxes):
        prob = sigmoid(np.array(pred_scores))
        num_queries = len(prob)
        num_gt = len(tgt_boxes)

        pos_cost = alpha * ((1 - prob) ** gamma) * -np.log(prob + 1e-8)
        neg_cost = (1 - alpha) * (prob ** gamma) * -np.log(1 - prob + 1e-8)
        cost_cls = pos_cost.reshape(-1, 1) - neg_cost.reshape(-1, 1)

        cost_l1 = np.zeros((num_queries, num_gt))
        for i, pb in enumerate(pred_boxes):
            for j, tb in enumerate(tgt_boxes):
                cost_l1[i, j] = np.abs(pb[0] - tb[0]) + np.abs(pb[1] - tb[1])

        def cw_to_t1t2(cw): return [cw[0] - cw[1]/2, cw[0] + cw[1]/2]
        cost_iou = np.zeros((num_queries, num_gt))
        for i, pb in enumerate(pred_boxes):
            ps, pe = cw_to_t1t2(pb)
            for j, tb in enumerate(tgt_boxes):
                ts, te = cw_to_t1t2(tb)
                inter = max(0, min(pe, te) - max(ps, ts))
                union = (pe - ps) + (te - ts) - inter
                cost_iou[i, j] = -inter / union if union > 0 else 0

        return lambda_cls * cost_cls + lambda_l1 * cost_l1 + lambda_iou * cost_iou

    for vid in vids:
        gt_segs = [a["segment"] for a in annotations[vid]["annotations"]]
        k = len(gt_segs)
        results = {}
        for name, model in zip(["TP²-DETR", "GAP", "Baseline"], [results_new, results_base, results_base2]):
            pred = model[vid]
            scores = [p["score"] for p in pred]
            segs = [p["segment"] for p in pred]
            centers = [(s[0] + s[1])/2 for s in segs]
            widths = [s[1] - s[0] for s in segs]
            pred_boxes = list(zip(centers, widths))

            gt_centers = [(s[0] + s[1])/2 for s in gt_segs]
            gt_widths = [s[1] - s[0] for s in gt_segs]
            tgt_boxes = list(zip(gt_centers, gt_widths))

            cost = build_cost_numpy(scores, pred_boxes, tgt_boxes)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_preds = [segs[i] for i in row_ind]
            avg_iou = np.mean([iou(gt_segs[j], matched_preds[i]) for i, j in enumerate(col_ind)])
            results[name] = (matched_preds, avg_iou)

        fig, ax = plt.subplots(figsize=(14, 2))
        for seg in gt_segs:
            ax.hlines(2, seg[0], seg[1], linewidth=8, label="Ground Truth" if seg == gt_segs[0] else '', colors='black')
            ax.vlines(seg[0], ymin=0, ymax=2, linestyles='dashed', linewidth=1, colors=(0, 0, 0, 0.1))
            ax.vlines(seg[1], ymin=0, ymax=2, linestyles='dashed', linewidth=1, colors=(0, 0, 0, 0.1))
        # for seg in results["GAP"][0]:
        #     ax.hlines(1.5, seg[0], seg[1], linewidth=8, label='GAP' if seg == results["GAP"][0][0] else '', colors=(0, 0, 0, 0.7))
        # for seg in results["Baseline"][0]:
        #     ax.hlines(1, seg[0], seg[1], linewidth=8, label='Baseline' if seg == results["Baseline"][0][0] else '', colors=(0, 0, 0, 0.3))
        for seg in results["GAP"][0]:
            rect = patches.Rectangle(
                (seg[0], 1.5 - 0.2),
                seg[1] - seg[0],
                0.3,
                linewidth=0,
                facecolor='lightgray',
                hatch='///////////',
                label='GAP' if seg == results["GAP"][0][0] else ''
            )
            ax.add_patch(rect)
        # Plot Baseline2 at y=1
        for seg in results["Baseline"][0]:
            rect = patches.Rectangle(
                (seg[0], 1 - 0.2),
                seg[1] - seg[0],
                0.3,
                linewidth=0,
                facecolor='gainsboro',
                hatch='\\\\\\\\\\\\',
                label='Baseline' if seg == results["Baseline"][0][0] else ''
            )
            ax.add_patch(rect)
        for seg in results["TP²-DETR"][0]:
            ax.hlines(0.5, seg[0], seg[1], linewidth=8, label='TP²-DETR' if seg == results["TP²-DETR"][0][0] else '', colors=(253/256, 222/256, 134/256))

        ax.set_yticks([0.5, 1, 1.5, 2])
        ax.set_yticklabels(['TP²-DETR', 'Baseline', 'GAP', 'Ground Truth'])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'(2) {vid} | k={k}, (AvgIoU: TP²-DETR {results["TP²-DETR"][1]:.2f} vs GAP {results["GAP"][1]:.2f} vs Baseline {results["Baseline"][1]:.2f})')
        ax.set_ylim(0, 2.5)
        ax.legend(loc=(1.02, 0))
        plt.tight_layout()
        plt.savefig(f'qualitative_iou_matching_{vid}.png')

if __name__=='__main__':
    iou_visualize()
    # recall_visualize()
    # scatter_visualize()
    # map_visualize()
    iou_matching_visualize()