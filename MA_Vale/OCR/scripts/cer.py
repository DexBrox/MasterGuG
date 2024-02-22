import timeit
start_eva = timeit.default_timer()

from Levenshtein import distance as lv_distance
from evaluate import load_files, calculate_polygon, calculate_midpoint, link_polygons_to_midpoints, sort_linked_data_by_polygon_and_midpoint_x, sum_sentences

output_dir = '../results'
gt_path = '../../labels'

def calculate_cer(sum_data):
    total_errors = 0 

    total_chars = sum(len(gt) for gt, _ in sum_data)

    for gt, pred in sum_data:
        dist = lv_distance(gt, pred)

        total_errors += dist

    cer = total_errors / total_chars if total_chars > 0 else 0
    return cer

def evaluate_cer(gt_path, pred_file_path):
    gt, pred = load_files(gt_path, pred_file_path)
    gt_poly, poly_only_text = calculate_polygon(gt)
    pred_mid, pred_mid_w_t = calculate_midpoint(pred)
    linked_data = link_polygons_to_midpoints(gt_poly, poly_only_text, pred_mid, pred_mid_w_t)
    sorted_data = sort_linked_data_by_polygon_and_midpoint_x(linked_data)
    sum_data = sum_sentences(sorted_data)

    cer_results = calculate_cer(sum_data)

    return cer_results

cer_result = evaluate_cer(gt_path, output_dir)
print(f"CER: {cer_result}")

end_eva = timeit.default_timer()
print(f"Evaluationszeit: {end_eva - start_eva} Sekunden")
