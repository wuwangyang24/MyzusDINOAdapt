import cv2
from PIL import Image
# from torchvision import transforms
import pandas as pd
import numpy as np
from collections import defaultdict
from glob import glob
# from UTILS.Anomaly_test.anomaly_score import Anomaly_scorer
import copy
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

## FLATTEN A MULTI-DIMENSIONAL LIST
def flatten_list(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)
    def __getitem__(self, key):
        return getattr(self, key)


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))

def log_state(state):
    result = []
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    return '\n'.join(result)


## GET THE ANGLE BY WHICH THE APHID SHOULD BE ROTATED TO FACE UPWARDS
def get_aligment(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection (example: Canny edge detection)
    edges = cv2.Canny(gray, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the object with length larger than width is the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Fit an ellipse to the contour to get its orientation
    ellipse = cv2.fitEllipse(largest_contour)
    major_axis_angle = ellipse[2]
    # Calculate the angle between the major axis and horizontal line
    angle = 180 - major_axis_angle
    return angle


## GET THE ANGLE BY WHICH THE APHID SHOULD BE ROTATED TO FACE UPWARDS
def get_aligment_from_image(pil_image):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)
    # Apply edge detection (example: Canny edge detection)
    edges = cv2.Canny(gray, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the object with length larger than width is the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Fit an ellipse to the contour to get its orientation
    ellipse = cv2.fitEllipse(largest_contour)
    major_axis_angle = ellipse[2]
    # Calculate the angle between the major axis and horizontal line
    angle = 180 - major_axis_angle
    return angle


## ALIGN APHIDS TO FACE UPWARDS
def align_image(image_path:str, image_size:int=96):
    image = Image.open(image_path)
    try:
        angle = get_aligment(image_path)
    except:
        angle = 0
    transform = transforms.Compose([
                transforms.Pad(((image_size-image.size[0])//2, 
                                (image_size-image.size[1])//2, 
                                image_size-image.size[0]-(image_size-image.size[0])//2, 
                                image_size-image.size[1]-(image_size-image.size[1])//2)),
                transforms.RandomRotation((-angle, -angle)),
                transforms.ToTensor()])
    return transform(image)


def align_image_from_image(image, image_size:int=96):
    try:
        angle = get_aligment_from_image(image)
    except:
        angle = 0
    transform = transforms.Compose([
                transforms.Pad(((image_size-image.size[0])//2, 
                                (image_size-image.size[1])//2, 
                                image_size-image.size[0]-(image_size-image.size[0])//2, 
                                image_size-image.size[1]-(image_size-image.size[1])//2)),
                transforms.RandomRotation((-angle, -angle)),
                transforms.ToTensor()])
    return transform(image)



## GET ALL INFO FOR A SUBSTANCE
## (BEST TO USE WITH PANDAS APPLY)
def get_sub_info(sub:str, metadata:dict, root_dir, plates=None):
    names = defaultdict(list)
    days = {}
    efficacies = {}
    dosages = {}
    image_paths = {}
    ref_plates = metadata.keys() if not plates else plates
    for plate in ref_plates:
        well_dosage = {}
        for i in range(len(metadata[plate]['data'])):
            pa_name = metadata[plate]['data'][i]['data']['properties']['myz:PA_Name']
            well = metadata[plate]['data'][i]
            if pa_name.find(sub)+1:
                # image names
                name = (well['plate']['vp_id'].replace('/', '_') + "_" + str(well['plate']['well']['x']) + "_" + str(well['plate']['well']['y']))
                names[plate].append(name)
                well_name = f'{plate}_{name}'
                # image paths
                image_paths[well_name] = glob(f'{root_dir}/Treated/{plate}/original/*{name}*.png')
                # days
                days[well_name] = int(well['data']['properties']['myz:Biolab_Evaluation_Time_Variable_Value'])
                # efficacies
                efficacies[well_name] = int(well['data']['classifier_results']['efficacy_auto'])
                # dosages 
                # if len(dosages)>0 and plate in [s.split('_')[0] for s in dosages.keys()]:
                #     print('##############')
                #     print(f'sub: {sub}')
                #     print(f'pa_name: {pa_name}')
                #     print(name)
                #     print(plate, float(metadata[plate]['data'][i]['data']['properties']['myz:Compound_Dosage']))
                #     print(dosages)
                #     print('##############')
                # else:
                dosages[well_name] = float(well['data']['properties']['myz:Compound_Dosage'])

    if len(names) == 0:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=['names', 'image_paths', 'days', 'efficacies', 'dosages'])
    return pd.Series([names, image_paths, days, efficacies, dosages], index=['names', 'image_paths', 'days', 'efficacies', 'dosages'])


## FILTER SUBSTANCES THAT HAVE CERTAIN DOSAGES EXPERIMENTED
## FOR EXAMPLE: DOSAGES [0.4, 8, 20, 100, 100]
def filter_subs_dosage(sub_table:pd.DataFrame, dosages:list):
    if_fit = sub_table.apply(lambda row: sorted(list(set(row['dosages'].values())), reverse=True)==dosages, axis=1)
    sub_table_unfit = sub_table[[i == 0 for i in if_fit]]
    sub_table_fit = sub_table[[i == 1 for i in if_fit]]
    return sub_table_unfit, sub_table_fit 

def concat_dicts(dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key].extend(value)  # Merge lists
            else:
                result[key] = value
    return result


## COMPUTE ANOMALY SCORES FOR A SUBSTANCE
def compute_anomaly_score(encoder, decoder, loss_function, image):
    if image is not None:
        encoded = encoder(image)
        decoded = decoder(encoded[1])
        loss = loss_function(decoded, image, encoded[1], encoded[2])
        return loss['Reconstruction_Loss'].detach().cpu().item(), loss['KLD'].detach().cpu().item()
    else:
        pass


def preprocess_single_image_for_inference(p, crop_size=(64, 96)):
    try:
        image = align_image_upwards(p)
        image = center_crop(image, crop_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        image = transform(image).unsqueeze(0)
        return image
    except:
        return None



## COMPUTE ANOMALY SCORES FOR A SUBSTANCE
def compute_z(encoder, image):
    encoded = encoder(image)
    return encoded[0].detach().cpu().tolist()


## COPOMPUTE THE AGGREGATED VALUE FOR ANOMALY SCORE FOR A SUBSTANCE
## IT CAN BE EITHER AGGREGATED BY DOSAGE OR BY NO ATTRIBUTES
def agg_anomaly_score(Ascores:dict, 
                      dosages:dict=None, 
                      mode:str='median', 
                      agg_all:bool=False, 
                      normalized:bool=False, 
                      ref_score:float=None, 
                      ref_score_dict:dict=None):
    if normalized:
        if ref_score:
            scores = {p:[d-ref_score for d in s] for p, s in Ascores.items()}
            missing_plates = []
        if ref_score_dict:
            ref_score = np.median(list(flatten_list(ref_score_dict.values())))
            scores = {p:[d-ref_score_dict[p.split('_')[0]] for d in s] if p.split('_')[0] in ref_score_dict.keys() else [0 for d in s] for p, s in Ascores.items()}
            missing_plates = [p for p, v in scores.items() if sum(v) == 0]
            # if len(missing_plates)>0:
            #     print(missing_plates)
    else:
        scores = copy.deepcopy(Ascores)
        missing_plates = []
    if not agg_all:
        merged_scores = {}
        for k, v in dosages.items():
            if k not in missing_plates:
                if v not in merged_scores:
                    merged_scores[v] = scores[k]
                if v in merged_scores:
                    merged_scores[v].extend(scores[k])
        if mode=='median':
            aggs = {p:np.median(merged_scores[p]) for p in merged_scores.keys()}
        elif mode=='mean':
            aggs = {p:np.mean(merged_scores[p]) for p in merged_scores.keys()}
        else:
            aggs = {p:np.sum(merged_scores[p]) for p in merged_scores.keys()}
        if len(aggs)>1:
            aggs = {key: aggs[key] for key in sorted(aggs)}
        else:
            aggs = np.nan
    else:
        if mode=='median':
            S = []
            for k,v in scores.items():
                if k not in missing_plates:
                    S.extend(v)
            if len(S)>0:
                aggs = np.median(S)
            else:
                aggs = np.nan
        elif mode=='mean':
            aggs = np.mean(list(flatten_list(scores.values())))
        else:
            aggs = np.sum(list(flatten_list(scores.values())))
    return aggs
    

## COMPUTE THE NORMALIZED AGGREGATED ANOMALY SCORE USING THE A REFERENCE SCORE
def normalize_scores(aggs, ref_score:float=None, ref_score_dict:dict=None):
    if ref_score:
        if type(aggs)==dict:
            responses = {p:aggs[p]-ref_score for p in aggs.keys()}
        else:
            responses = aggs-ref_score
    if ref_score_dict:
        if type(aggs)==dict:
            responses = {p:aggs[p]-ref_score for p in aggs.keys()}
        else:
            responses = aggs-ref_score

    return responses


## FIT A CURVE TO THE ANOMALY SCORES CORRESPONDING TO DIFFERENT DOSAGES
## CURVE TYPE: LINEAR OR SIGMOID
def fit_dose_response_curve(dose_score:dict, curve_type:str='linear'):
    dose_score = {key: dose_score[key] for key in sorted(dose_score.keys()) if dose_score[key]!=np.nan}
    X = list(dose_score.keys())
    y = list(flatten_list(dose_score.values()))
    try:
        if curve_type=='linear':
            slope, intercept = np.polyfit(X, y, 1)
            return slope, intercept
        else:
            raise ValueError("curve type can either be linear or sigmoid") 
    except Exception as e:
        print(e)
        return np.nan


## GET THE TOP X PERENTILE OF THE ANOMALY SCORES OF A CERTAIN SUBSTANCE 
## RETURN A DICT CONTAINING ANOMALY SCORES, CORRESPONDING DOSAGES AND EFFICACIES
def get_top_percentile(anomaly_scores:dict, dosages:dict, percentile:int=90, mode:str='mean'):
    top_percentile = {}
    for k, v in anomaly_scores.items():
        if len(v)>0:
            percentile_thread = np.percentile(sorted(v), percentile)
            percentile_scores = [num for num in v if num >= percentile_thread]
            dosage = dosages[k]
            if k in top_percentile:
                top_percentile[dosage].extend(percentile_scores)
            else:
                top_percentile[dosage] = percentile_scores
        else:
            dosage = dosages[k]
            top_percentile[dosage] = np.nan
    return {key: np.mean(top_percentile[key]) if mode=='mean' else np.median(top_percentile[key]) for key in sorted(top_percentile)}


## GET THE TOP X PERENTILE OF THE ANOMALY SCORES OF A CERTAIN SUBSTANCE 
## RETURN A DICT CONTAINING ANOMALY SCORES, CORRESPONDING DOSAGES AND EFFICACIES
def get_top_percentile_sub(anomaly_scores:dict, percentile:int=90, mode:str='median', normalize=False, control_score=None):
    scores = list(flatten_list(anomaly_scores.values()))
    if len(scores)>1:
        percentile_thread = np.percentile(sorted(scores), percentile)
        percentile_scores = [num for num in scores if num >= percentile_thread]
        if not normalize:
            return np.mean(percentile_scores) if mode=='mean' else np.median(percentile_scores)
        else:
            return np.mean(percentile_scores)-control_score if mode=='mean' else np.median(percentile_scores)-control_score
    else:
        return np.nan



## CALCULATE THE MEAN EFFICACY AT EACH DOSAGE FOR EACH SUB
def eff_at_dosage(efficacies:dict, dosages:dict):
    eff_dose = {}
    for d in list(set(dosages.values())):
        eff_dose[d] = np.mean([efficacies[p] for p in efficacies.keys() if dosages[p]==d])
    return {key: eff_dose[key] for key in sorted(eff_dose)}



## CALCULATE THE MEAN EFFICACY FOR EACH SUB
def eff_at_sub(efficacies:dict):
    return np.mean(list(flatten_list(efficacies.values())))


## COUNT THE NUMBER OF PIXELS IN AN IMAGE
def count_pixel(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_array = np.array(image)
    non_black_pixels = np.sum((image_array[:, :, 0] != 0) | (image_array[:, :, 1] != 0) | (image_array[:, :, 2] != 0))
    return non_black_pixels

## GET THE COORDINATES OF THE EDGE
def extract_edge_pixel_coors(image_path, align=False, rotate=False):
    if not align:
        image = cv2.imread(image_path)
    else:
        image = align_image(image_path)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8) 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    images = [image]
    if rotate:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        images.append(rotated_image)
    results = {}
    for idx, img in enumerate(images):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_image = np.zeros_like(image)
        cv2.drawContours(outline_image, contours, -1, (255, 255, 255), thickness=1)
        outline_dict = {}
        for x in range(outline_image.shape[0]):
            for y in range(outline_image.shape[1]):
                pixel_value = outline_image[x, y]==[0,0,0]
                if not np.all(pixel_value):
                    outline_dict[str((x, y))] = int(not np.all(pixel_value))
        results[idx] = outline_dict
    return results

## GET COLOR DISTRIBUTION
def get_color_dist(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    object_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    hist_r = cv2.calcHist([object_pixels], [0], mask, [256], [0, 256])
    hist_g = cv2.calcHist([object_pixels], [1], mask, [256], [0, 256])
    hist_b = cv2.calcHist([object_pixels], [2], mask, [256], [0, 256])
    color_distribution = {}
    for c, h in zip(['r', 'g', 'b'], [hist_r, hist_g, hist_b]):
        color_distribution[c] = [float(i[0]) for i in h]
    return color_distribution


def align_image_upwards(image_path, return_edge=False):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        rect_width, rect_height = rect[1] 
        center = rect[0]
        center_x = center[0]
        center_y = center[1]
        w,h = image.shape[1], image.shape[0]
        new_width = 96
        new_height = 96
        pad_x_left = int(max(0, new_width// 2 - center_x))
        pad_x_right = int(max(0, new_width - w - pad_x_left))
        pad_y_top = int(max(0, new_height// 2 - center_y))
        pad_y_bottom = int(max(0, new_height - h - pad_y_top))
        padded_image = cv2.copyMakeBorder(image, pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        print("No object found in the image.")
    (h, w) = padded_image.shape[:2]
    angle = get_aligment(image_path)
    center = (w // 2, h // 2)
    upper_y = center[1]-rect_height/2
    bottom_y = center[1]+rect_height/2
    left_x = center[0]-rect_width/2
    right_x = center[0]+rect_width/2
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # The last parameter is the scale
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    darkest_pixel = None
    min_intensity = 255  # Initialize with the maximum possible intensity
    for contour in contours:
        # Create a mask for the current contour
        contour_mask = np.zeros_like(gray_image)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
        pixels = gray_image[contour_mask == 255]
        if len(pixels) > 0:
            min_val = np.min(pixels)
            if min_val < min_intensity:
                min_intensity = min_val
                darkest_pixel = np.argwhere(gray_image == min_intensity)[0]  # Get coordinates

    # if darkest point is in the upper part and more closer to the center
    # then head is wrongly oriented
    if darkest_pixel[0]<48 and (darkest_pixel[0]-upper_y)>(48-darkest_pixel[0]):
        angle = 180
    # if darkest point is in the lower part and closer to the lower border
    # then it needs to be rotated too
    elif darkest_pixel[0]>48 and (bottom_y-darkest_pixel[0])<(darkest_pixel[0]-48):
        angle = 180
    else:
        angle = 0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # The last parameter is the scale
    rotated_image_final = cv2.warpAffine(rotated_image, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    if return_edge:
        gray = cv2.cvtColor(rotated_image_final, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        return edges
    return rotated_image_final


def center_crop(image, crop_size):
        height, width, _ = image.shape
        start_x = (width - crop_size[0]) // 2
        start_y = (height - crop_size[1]) // 2
        cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
        return cropped_image


## FUNCTIONS
def sort_row(row, ref_col):
    indices = sorted(range(len(row[ref_col])), key=lambda i: row[ref_col][i], reverse=True)
    cols_to_sort = row.keys()[1:]
    for col in cols_to_sort:
        row[col] = np.array(row[col])[indices].tolist()
    return row

def filter_dosage_near_hit_eff(row, eff_ref):
    dosages = row['dosage']
    effs = row['efficacy']
    index = []
    previous_index = None
    unique_dosages = list(np.unique(dosages))
    unique_dosages.sort(reverse=True)
    for idx, d in enumerate(unique_dosages):
        indices = [i for i, value in enumerate(dosages) if value == d]
        es = [effs[i] for i in indices]
        flag = any(e >= eff_ref for e in es)
        if flag:
            #index = indices[-1]+1 if index is None or index < len(effs)-1 else index
            previous_index = indices
        else:
            if previous_index is not None:
                index.extend(previous_index)
            index.extend(indices)
            break
    cols_to_cut = row.keys()[1:]
    if index is not None:
        for col in cols_to_cut:
            row[col] = [row[col][i] for i in index]
    return row


def get_scores(row, anomaly_scores_ref):
    hits_scores = {}
    hits_paths = {}
    hits_missing_plates = []
    for idx, plate in enumerate(row.plate):
        well = f'_{row.well_x[idx]}_{row.well_y[idx]}_bug'
        try:
            scores = anomaly_scores_ref[str(plate)]
            filtered_scores = [scores['scores'][i] for i in range(len(scores['paths'])) if well in scores['paths'][i]]
            filtered_paths = [scores['paths'][i] for i in range(len(scores['paths'])) if well in scores['paths'][i]]
            hits_scores[f'{plate}_{row.well_x[idx]}_{row.well_y[idx]}'] = filtered_scores
            hits_paths[f'{plate}_{row.well_x[idx]}_{row.well_y[idx]}'] = filtered_paths
        except:
            hits_missing_plates.append(plate)
    return hits_scores, hits_paths, hits_missing_plates

def get_paths(row, root):
    paths = {}
    for idx, plate in enumerate(row.plate):
        well = f'_{row.well_x[idx]}_{row.well_y[idx]}_bug'
        candidate_paths = glob(f'{root}/{plate}/original/**')
        paths[f'{plate}_{row.well_x[idx]}_{row.well_y[idx]}'] = [p for p in candidate_paths if well in p]
    return paths



def get_score_at_dosage(row, agg=np.median, percentile=None, cut_percentile=None):
    score_at_dosage = defaultdict(list)
    for d, p, x, y in zip(row['dosage'], row['plate'], row['well_x'], row['well_y']):
        try:
            well = f'{p}_{x}_{y}'
            score_at_dosage[d].extend(row['scores'][well])
            score_at_dosage['all'].extend(row['scores'][well])
        except:
            pass
    if not cut_percentile:
        if not percentile:
                score_at_dosage = {k:agg(v) for k,v in score_at_dosage.items() if len(v)>0}
        else:
            score_at_dosage = {k:np.percentile(v, percentile) for k,v in score_at_dosage.items() if len(v)>0}
    else:
        thres = {k:np.percentile(V, cut_percentile) for k, V in score_at_dosage.items() if len(V)>0}
        score_at_dosage = {k:agg([v for v in V if v>=thres[k]]) for k,V in score_at_dosage.items() if len(V)>0}
    return score_at_dosage

def get_efficacy_at_dosages(row, hit=True):
    effs = defaultdict(list)
    for idx , d in enumerate(row['dosage']):
        try:
            effs[d].append(row['efficacy'][idx])
            # effs['all'].append(row['efficacy'][idx])
        except:
            pass
    effs = {k:int(np.mean(v)) if max(v)<70 else max(int(np.mean(v)), 70) for k,v in effs.items() if len(v)>0}
    # else:
    #     effs = {k:int(np.mean(v)) for k,v in effs.items() if len(v)>0}
    return effs


def get_size(row):
    size_dict = {}
    paths = row['paths']
    for idx, (p, x, y) in enumerate(zip(row['plate'], row['well_x'], row['well_y'])):
        k = f'{p}_{x}_{y}'
        S = []
        try:
            for p in row['paths'][k]:
                S.append(int(count_pixel(p)))
            size_dict[k] = S
        except:
            pass
    return size_dict

def get_color(row):
    color_dict = {}
    paths = row['paths']
    for idx, (p, x, y) in enumerate(zip(row['plate'], row['well_x'], row['well_y'])):
        k = f'{p}_{x}_{y}'
        color_hist = []
        try:
            for p in row['paths'][k]:
                c_dict = get_color_dist(p)
                color_hist.append(c_dict)
            color_dict[k] = color_hist
        except:
            pass
    return color_dict 

def get_shape(row):
    shape_dict = {}
    paths = row['paths']
    for idx, (p, x, y) in enumerate(zip(row['plate'], row['well_x'], row['well_y'])):
        k = f'{p}_{x}_{y}'
        S = []
        try:
            for p in row['paths'][k]:
                shape = align_image_upwards(p, True)
                S.append(np.column_stack(np.where(shape==255)).tolist())
            shape_dict[k] = S
        except:
            pass
    return shape_dict

def process_row(row, root):
    root_hits = 'compound_profile/hits'
    patch_no = row['preparation_no']
    profile = {}
    profile['color'] = get_color(row)
    profile['shape'] = get_shape(row)
    profile['size'] = get_size(row)
    with open(f'{root}/{patch_no}.json', 'w') as json_file:
        json.dump(profile, json_file, indent=4)


def extract_scores_at_dosage(df, feature, dosage, percentile=None):
    scores_dosage = []
    for _, row in df.iterrows():
        try:
            for i in range(len(row['dosage'])):
                if row['dosage'][i]==dosage:
                    if percentile:
                        threshold = np.percentile(row[feature][dosage], percentile)
                        scores_dosage.extend([item for item in row[feature][dosage] if item > threshold])
                    else:
                        scores_dosage.append(row[feature][dosage])
        except:
            pass
    return scores_dosage


def extract_size_at_dosage(df, dosage, percentile=None):
    size_dosage = []
    for _, row in df.iterrows():
        preparation_no = row['preparation_no']
        try:
            with open(f'compound_profile/hits/{preparation_no}.json', 'r') as file:
                profile = json.load(file)
            profile_size = profile['size']
            for i in range(len(row['dosage'])):
                if row['dosage'][i]==dosage:
                    well = f"{row['plate'][i]}_{row['well_x'][i]}_{row['well_y'][i]}"
                    size = profile_size[well]
                    if percentile:
                        threshold = np.percentile(size, percentile)
                        size_dosage.extend([item for item in size if item > threshold])
                    else:
                        size_dosage.extend(size)
        except Exception as e:
            pass
    return size_dosage

def extract_color_at_dosage(df, dosage, color='r'):
    color_dosage = []
    for _, row in df.iterrows():
        try:
            for i in range(len(row['dosage'])):
                if row['dosage'][i]==dosage:
                    color_dosage.extend(row['color'][i][color])
        except:
            pass
    return color_dosage

def plot_feature_dosage(feature, hits, nohits, agg=np.median):
    medians_hits = []
    medians_nohits = []
    feature_scores_hits = []
    feature_scores_nohits = []
    F = extract_size_at_dosage if 'size' in feature else extract_scores_at_dosage
    for dosage in [0.8, 4, 20]:
        feature_hits = F(hits, feature, dosage)
        feature_scores_hits.append(feature_hits)
        feature_nohits = F(nohits, feature, dosage)
        feature_scores_nohits.append(feature_nohits)
        medians_hits.append(np.median(feature_hits))
        medians_nohits.append(np.median(feature_nohits))

    fig = go.Figure()
    X = [0.8, 4, 20]
    fig.add_trace(go.Scatter(y=medians_hits, x=X, mode='lines+markers', name='Hits'))
    fig.add_trace(go.Scatter(y=medians_nohits, x=X, mode='lines+markers', name='No Hits'))
    title = f'Median {feature}'
    fig.update_layout(title=title,
                    xaxis_title='Dosage',
                    yaxis_title=feature)
    fig.show()


    for idx, x in enumerate(X):
        fig = go.Figure()
        feature_hits = feature_scores_hits[idx]
        feature_nohits = feature_scores_nohits[idx]
        fig.add_trace(go.Histogram(x=feature_hits, name='Hits', opacity=0.75, histnorm='probability density', xbins=dict(start=min(feature_hits), 
                                            end=max(feature_hits), 
                                            size=(max(feature_hits) - min(feature_hits)) / 5000)))
        fig.add_trace(go.Histogram(x=feature_nohits, name='No Hits', opacity=0.75, histnorm='probability density', xbins=dict(start=min(feature_nohits), 
                                            end=max(feature_nohits), 
                                            size=(max(feature_nohits) - min(feature_nohits)) / 5000)))
        title = f'Median {feature} at {x} ppm'
        fig.update_layout(title=title,
                            xaxis_title='Dosage',
                            yaxis_title=feature)
        fig.show()


def check_size(row, dosage=None):
    preparation_no = row['preparation_no']
    with open(f'compound_profile/nohits/{preparation_no}.json', 'r') as file:
        profile = json.load(file)
    sizes = profile['size']
    if not dosage:
        return list(itertools.chain.from_iterable(sizes.values()))


def check_shape(row, dosage=None):
    preparation_no = row['preparation_no']
    with open(f'compound_profile/nohits/{preparation_no}.json', 'r') as file:
        profile = json.load(file)
    shapes = profile['shape']
    shape_coors = defaultdict(int)
    for w, S in shapes.items():
        for s in S:
            for i in s:
                shape_coors[(i[0], i[1])] += 1
    return shape_coors


def plot_sizes(criteria, smoothing=0.05):
    if criteria == 'max':
        F = np.max
    elif criteria == 'min':
        F = np.min
    elif criteria == 'mean':
        F = np.mean
    elif criteria == 'median':
        F = np.median
    elif criteria == 'max-min':
        def F(l):
            return np.max(l)-np.min(l)
    else:
        return 'wrong criteria'
    sizes = [F(check_size(nomination.iloc[i,])) for i in range(len(nomination))]
    def exponential_moving_average(data, alpha):
        ema = [data[0]]
        for price in data[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        return ema
    ema_sizes = exponential_moving_average(sizes, smoothing)
    plt.figure(figsize=(12, 4))
    plt.plot(sizes, label='Original Sizes', alpha=0.5)
    plt.plot(ema_sizes, label='Exponential Moving Average', color='orange')
    plt.title(f'{criteria} Size')
    plt.xlabel('Rank')
    plt.ylabel('Size')
    plt.legend()
    plt.show()


# def bin_eff(y_eff, thres1=40, thres2=70):
#     y_bin = []
#     for number in y_eff:
#         if number < thres1:
#             y_bin.append(0)
#         elif number >=thres1 and number<thres2:
#             y_bin.append(1)
#         else:
#             y_bin.append(2)
#     return y_bin

def bin_eff(y_eff, thresholds):
    y_bin = []
    thresholds = sorted(thresholds)
    for number in y_eff:
        bin_index = 0
        for i, thres in enumerate(thresholds):
            if number < thres:
                break
            bin_index = i + 1
        y_bin.append(bin_index)
    return y_bin


def expand_dataframe(df):   
    expanded_df = []
    for _, row in df.iterrows():
        for i in range(len(row.dosage)):
            if f'{row.plate[i]}_{row.well_x[i]}_{row.well_y[i]}' not in row.paths:
                paths = []
            else:
                paths = row.paths[f'{row.plate[i]}_{row.well_x[i]}_{row.well_y[i]}']
            expanded_df.append({'preparation_no':row.preparation_no, 'dosage':row.dosage[i], 'plate':row.plate[i], 'well_x':row.well_x[i], 'well_y':row.well_y[i], 'paths':paths})
    return pd.DataFrame(expanded_df)


def extract_paths(df_meta):
    dict_paths = {}
    for idx, row in df_meta.iterrows():
        sub = row.preparation_no
        sub_dict = defaultdict(list)
        for i, d in enumerate(row.dosage):
            try:
                paths = row.paths[f'{row.plate[i]}_{row.well_x[i]}_{row.well_y[i]}']
                sub_dict[d].extend(paths)
            except:
                pass
        dict_paths[sub] = dict(sub_dict)
    return dict_paths