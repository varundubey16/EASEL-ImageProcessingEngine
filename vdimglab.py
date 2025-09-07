"""
22-10-24 starting
v1.0

current - v4.17


new pips needed-    pip install 
flask opencv-python numpy pillow scikit-image scikit-learn + pytesseract + need tesseract OCR instld on comp.
"""
import uuid
import math
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage import data, exposure, feature
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, session, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
warnings.filterwarnings("ignore")
apptitle = "EASELbyVarun"
def touint8(image):#Convert img array to uint8
    arr = np.asarray(image)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    if arr.dtype == bool:
        return (arr.astype(np.uint8) * 255)
    arr = np.nan_to_num(arr)
    if arr.size and arr.max() <= 1.0 and arr.min() >= 0:
        arr = arr * 255
    return np.clip(arr, 0, 255).astype(np.uint8)
def torgb(image):
    arr = touint8(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError("Please use a standard grayscale or RGB image.")
def togray(image):
    arr = touint8(image)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(torgb(arr), cv2.COLOR_RGB2GRAY)
def oddkrnl(value):
    value = max(1, int(round(value)))
    return value if value % 2 else value + 1
'''def touint8(image):
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] != 3):
        raise ValueError("Expected grayscale, RGB or RGBA image.")
    if arr.dtype == np.uint8:
        return arr.copy()
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0, posinf=255, neginf=0)
    if arr.size and arr.min() >= 0 and arr.max() <= 1:
        arr *= 255
    return np.clip(arr, 0, 255).astype(np.uint8)'''    
workspaces = {
    "Universal Lab": [],
    "Image Enhancement": [
        "Brightness & contrast", "Gamma correction", "Histogram equalization",
        "CLAHE local contrast", "Laplacian sharpening (class practical)",
        "Min-max normalization", "Unsharp mask",
    ],
    "Image Filters": [
        "Average blur", "Gaussian blur", "Median filter", "Minimum filter (class practical)",
        "Maximum filter (class practical)", "Bilateral filter", "Sharpen filter", "Emboss filter",
    ],
    "Edge Detection": [
        "Sobel magnitude (class practical)", "Sobel X", "Sobel Y", "Prewitt edges",
        "Scharr edges", "Laplacian edges", "Canny edges", "Roberts edges",
    ],
    "Segmentation": [
        "Global threshold", "Otsu threshold", "Adaptive threshold",
        "HSV colour segmentation", "Horizontal projection / line segmentation",
        "Vertical projection / word segmentation", "Character segmentation",
        "Connected-component segmentation", "Watershed segmentation",
    ],
    "Morphology": [
        "Erosion", "Dilation", "Opening", "Closing", "Morphological gradient",
        "Top-hat", "Black-hat",
    ],
    "Contours & Chain Codes": [
        "Draw external contours", "Contour bounding boxes", "Contour measurements",
        "4-directional chain code (class practical)",
        "8-directional chain code (class practical)", "Object count",
    ],
    "OCR & Handwriting": [
        "Document binarization (Otsu)", "Deskew document", "Line segmentation",
        "Word segmentation", "Character segmentation", "OCR text extraction (Tesseract)",
        "Handwritten digit recognition (SVM)", "Handwritten digit recognition (KNN)",
    ],
    "Medical Image Analysis": [
        "Medical contrast enhancement (CLAHE)", "Medical denoising (median)",
        "Medical boundary extraction (Canny)", "Medical region segmentation (Otsu)",
        "Medical connected-region measurements",
    ],
    "Industrial Vision": [
        "Component counting", "Shape classification", "Surface edge inspection",
        "Line detection (Hough)", "ORB feature inspection", "Template matching",
    ],
    "Sketch & Blueprint": [
        "Pencil sketch", "Canny line art", "Blueprint threshold",
        "Line detection (Hough)", "Contour drawing",
    ],
    "Colour & Pattern": [
        "RGB channel — Red", "RGB channel — Green", "RGB channel — Blue",
        "HSV channel — Hue", "HSV channel — Saturation", "HSV channel — Value",
        "HSV colour segmentation", "Dominant-colour palette (K-means)",
        "K-means colour quantization",
    ],
    "Computer Vision & Classification": [
        "ORB keypoints", "HOG feature visualization", "Hu shape moments",
        "Shape classification", "Template matching",
        "Handwritten digit recognition (SVM)", "Handwritten digit recognition (KNN)",
    ],
    "Image Transformations": [
        "Rotate 90° clockwise", "Rotate 90° counter-clockwise", "Flip horizontally",
        "Flip vertically", "Resize to 50%", "Resize to 150%",
    ],
    "Frequency Domain": [
        "Fourier magnitude spectrum", "Fourier low-pass filter", "Fourier high-pass filter",
    ], 
}
allops = list(dict.fromkeys(op for ops in workspaces.values() for op in ops))
def makemask(gray, threshold=127, invert=True):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, int(threshold), 255, mode)
    return out
def groupruns(active):#return halfopen rngs for contiguous True vals
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, split_points)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups if len(group)]
def largestcontours(binary, minarea=20):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= float(minarea)]
def contourovl(image, minarea=20, boxes=False, labels=False):
    rgb = torgb(image)
    gray = togray(rgb)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = largestcontours(binary, minarea)
    output = rgb.copy()
    for i, contour in enumerate(contours, 1):
        cv2.drawContours(output, [contour], -1, (40, 220, 90), 2)
        if boxes or labels:
            x, y, w, h = cv2.boundingRect(contour)
            if boxes:
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 80, 40), 2)
            if labels:
                cv2.putText(output, str(i), (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 40), 2)
    return output, contours, binary
def chaincode(contour, connectivity=8):
    points = contour.reshape(-1, 2)
    if len(points) < 2:
        return []
    codes = []
    dirs8 = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]#directions=clockwise, start right. coords(dx, dy).
    dirs4 = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    lookup8 = {d: i for i, d in enumerate(dirs8)}
    lookup4 = {d: i for i, d in enumerate(dirs4)}
    for p1, p2 in zip(points[:-1], points[1:]):
        dx, dy = int(p2[0] - p1[0]), int(p2[1] - p1[1])
        if connectivity == 8:
            if (dx, dy) in lookup8:
                codes.append(lookup8[(dx, dy)])
        else:
            sx = 1 if dx > 0 else -1 if dx < 0 else 0
            sy = 1 if dy > 0 else -1 if dy < 0 else 0# Convert diagonal 8-neighbour moves into two 4-neighbour moves.
            if sx:
                codes.append(lookup4[(sx, 0)])
            if sy:
                codes.append(lookup4[(0, sy)])
    return codes
def regiontbl(binary, minarea=20):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < minarea:
            continue
        rows.append({"label": idx, "area": int(area), "centroid": (round(float(centroids[idx][0]), 1), round(float(centroids[idx][1]), 1)), "bbox": (int(x), int(y), int(w), int(h))})
    return labels, rows
digitmodels = None
digitinfo = None
def getdigitmodels():
    global digitmodels, digitinfo
    if digitmodels is None:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
        )
        svm_model = SVC(kernel="rbf", gamma=0.001)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        svm_model.fit(X_train, y_train)
        knn_model.fit(X_train, y_train)
        digitmodels = {"SVM": svm_model, "KNN": knn_model}
        digitinfo = {
            "SVM": accuracy_score(y_test, svm_model.predict(X_test)),
            "KNN": accuracy_score(y_test, knn_model.predict(X_test)),
        }
    return digitmodels, digitinfo
def recogdigit(image, model_name):
    gray = togray(image)#find the likely foreground. Handwritten digits are commonly dark on light paper.
    if np.mean(gray) > 127:
        foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(foreground > 0)
    if len(xs) == 0:
        raise ValueError("No foreground digit was found. Try a clearer, tightly cropped digit image.")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = foreground[y0:y1, x0:x1]
    side = max(crop.shape) + 8 # Fit inside a square and centre on an 8x8 canvas like sklearn's digits dataset.
    canvas = np.zeros((side, side), dtype=np.uint8)
    yoff = (side - crop.shape[0]) // 2
    xoff = (side - crop.shape[1]) // 2
    canvas[yoff:yoff + crop.shape[0], xoff:xoff + crop.shape[1]] = crop
    resized = cv2.resize(canvas, (8, 8), interpolation=cv2.INTER_AREA)
    features = (resized.astype(np.float32) / 255.0 * 16.0).reshape(1, -1)
    models, info = getdigitmodels()
    prediction = int(models[model_name].predict(features)[0])
    details = (
        f"Predicted digit: {prediction}\n"
        f"Classifier: {model_name}\n"
        f"Held-out accuracy on sklearn's built-in digits dataset: {info[model_name] * 100:.1f}%\n\n"
        "This is a small demonstration classifier, not a production handwriting OCR model. "
        "It expects one reasonably centred digit."
    )
    return cv2.resize(resized, (240, 240), interpolation=cv2.INTER_NEAREST), details

def deskew(image):
    rgb = torgb(image)
    gray = togray(rgb)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return rgb, "Not enough foreground pixels to estimate skew; image left unchanged."
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    corrected = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return corrected, f"Estimated skew correction: {angle:.2f} degrees."
'''
def projsegments(image, axis="horizontal", minarea=20):
    rgb = torgb(image)
    gray = togray(rgb)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if axis == "horizontal":
        projection = np.sum(binary > 0, axis=1)
        active = projection > max(1, int(binary.shape[1] * 0.005))
        runs = [(s, e) for s, e in groupruns(active) if e - s > 1]
        output = rgb.copy()
        for i, (start, end) in enumerate(runs, 1):
            cv2.rectangle(output, (0, start), (rgb.shape[1] - 1, end - 1), (30, 220, 80), 2)
            cv2.putText(output, f"Line {i}", (6, min(end + 18, rgb.shape[0] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 70, 40), 1)
        return output, f"Detected {len(runs)} horizontal line band(s) using horizontal projection."
    elif axis == "vertical":
        projection = np.sum(binary > 0, axis=0)
        active = projection > max(1, int(binary.shape[0] * 0.005))
        runs = [(s, e) for s, e in groupruns(active) if e - s > 1]
        output = rgb.copy()
        for start, end in runs:
            cv2.rectangle(output, (start, 0), (end - 1, rgb.shape[0] - 1), (30, 220, 80), 2)
        return output, f"Detected {len(runs)} vertical word/character group(s) using vertical projection."
    else:
        raise ValueError("Axis must be 'horizontal' or 'vertical'.")
def watershed(image, minarea=20):
    rgb = torgb(image)
    gray = togray(rgb)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    background = cv2.dilate(opening, kernel, iterations=3)
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if distance.max() == 0:
        return rgb, "No foreground objects found. Try improving image contrast."
    foreground = cv2.threshold(distance, 0.35 * distance.max(), 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    unknown = cv2.subtract(background, foreground)
    _, markers = cv2.connectedComponents(foreground)
    markers += 1
    markers[unknown > 0] = 0
    markers = cv2.watershed(rgb.copy(), markers)
    output = rgb.copy()
    output[markers == -1] = (255, 40, 40)
    count = max(0, len(np.unique(markers)) - 2)
    return output, f"Watershed detected approximately {count} foreground region(s). Boundaries are marked in red."
def shapeclass(image, minarea=20):
    output, contours, binary = contourovl(image, minarea, boxes=True, labels=False)
    counts = {"Triangle": 0, "Quadrilateral": 0, "Circle / polygon": 0}
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        if vertices == 3:
            name = "Triangle"
        elif vertices == 4:
            name = "Quadrilateral"
        else:
            name = "Circle / polygon"
        counts[name] += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(output, name, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 40), 1)
    details = "Shape classification (contour approximation):\n" + "\n".join(f"{name}: {count}" for name, count in counts.items())
    return output, details
def houghlines(image, threshold=127):
    rgb = torgb(image)
    gray = togray(rgb)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(10, int(threshold)), minLineLength=25, maxLineGap=10)
    output = rgb.copy()
    count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 40, 40), 2)
            count += 1
    return output, f"Detected {count} line segment(s) using the probabilistic Hough transform."
'''
def projsegments(image, axis="horizontal", minarea=20):
    rgb = torgb(image)
    gray = togray(rgb)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if axis == "horizontal":
        projection = np.sum(binary > 0, axis=1)
        active = projection > max(0, int(binary.shape[1] * 0.005))
        runs = groupruns(active)
        output = rgb.copy()
        for i, (start, end) in enumerate(runs, 1):
            if end - start <= 1:
                continue
            cv2.rectangle(output, (0, start), (rgb.shape[1] - 1, end - 1), (30, 220, 80), 2)
            cv2.putText(output, f"Line {i}", (6, min(end + 18, rgb.shape[0] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 70, 40), 1)
        return output, f"Detected {len(runs)} horizontal line band(s) using a horizontal projection profile."
    projection = np.sum(binary > 0, axis=0)
    active = projection > max(0, int(binary.shape[0] * 0.005))
    runs = [(s, e) for s, e in groupruns(active) if e - s > 1]
    output = rgb.copy()
    for i, (start, end) in enumerate(runs, 1):
        cv2.rectangle(output, (start, 0), (end - 1, rgb.shape[0] - 1), (30, 220, 80), 2)
    label = "word/character column group(s)"
    return output, f"Detected {len(runs)} vertical {label} using a vertical projection profile. " \
        "For multi-line documents, first run line segmentation or crop one line for better word/character separation."
def watershed(image, minarea=20):
    rgb = torgb(image)
    gray = togray(rgb)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return rgb, "Watershed could not find foreground objects in this image. Try a clearer image or different contrast."
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(rgb.copy(), markers)
    output = rgb.copy()
    output[markers == -1] = (255, 40, 40)
    count = len(set(np.unique(markers)) - {-1, 1})
    return output, f"Watershed marked boundaries in red. Approximate foreground regions: {count}."
def shapeclass(image, minarea=20):
    output, contours, binary = contourovl(image, minarea, boxes=True, labels=False)
    counts = {"Triangle": 0, "Quadrilateral": 0, "Circle / polygon": 0}
    for i, contour in enumerate(contours, 1):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        if vertices == 3:
            name = "Triangle"
        elif vertices == 4:
            name = "Quadrilateral"
        else:
            name = "Circle / polygon"
        counts[name] += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(output, name, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 40), 1)
    details = "Contour-based shape estimate (heuristic, not a trained model):\n" + "\n".join(f"{k}: {v}" for k, v in counts.items())
    return output, details
def houghlines(image, threshold=127):
    gray = togray(image)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(10, int(threshold)), minLineLength=25, maxLineGap=10)
    output = torgb(image).copy()
    count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 40, 40), 2)
            count += 1
    return output, f"Detected {count} line segment(s) using the probabilistic Hough transform."
def templatematch(image, templateimg, threshold=0.75):
    if templateimg is None:
        raise ValueError("Upload a template image in the optional Template image box before running Template matching.")
    scene = togray(image)
    template = togray(templateimg)
    h, w = template.shape
    if h > scene.shape[0] or w > scene.shape[1]:
        raise ValueError("The template image must be smaller than the main image.")
    result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= float(threshold) / 100.0)#reduce duplicate overlapping locs by sorting by score + suppressing nearby boxes.
    candidates = sorted([(float(result[y, x]), int(x), int(y)) for y, x in zip(*loc)], reverse=True)
    chosen = []
    '''
    for score, x, y in candidates:
    duplicate = any(abs(x - xx) <= w // 2 and abs(y - yy) <= h // 2 for _, xx, yy in chosen)
    if not duplicate:
        chosen.append((score, x, y))
    if len(chosen) >= 30:
        break
    output = torgb(image).copy()
    for score, x, y in chosen:
        cv2.rectangle(output, (x, y), (x + w, y + h), (40, 220, 60), 2)
    return output, f"Template matches above {threshold:.0f}% similarity: {len(chosen)}. Template size: {w} × {h}px."
    '''
    for score, x, y in candidates:
        if all(abs(x - xx) > w // 2 or abs(y - yy) > h // 2 for _, xx, yy in chosen):
            chosen.append((score, x, y))
        if len(chosen) >= 30:
            break
    output = torgb(image).copy()
    for score, x, y in chosen:
        cv2.rectangle(output, (x, y), (x + w, y + h), (40, 220, 60), 2)
    return output, f"Template matches above {threshold:.0f}% similarity: {len(chosen)}. Template size: {w} × {h}px."
def domcolors(image, k=5, quantize=False):
    rgb = torgb(image)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (min(250, w), min(250, h)), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3)).astype(np.float32)
    k = max(2, min(int(k), 8, len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    if quantize:
        labels_full = cv2.resize(labels.reshape(small.shape[:2]).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        output = centers[labels_full]
    else:
        output = np.full((140, 420, 3), 245, dtype=np.uint8)
        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(counts)[::-1]
        x = 0
        for idx in order:
            width = max(1, int(420 * counts[idx] / len(labels)))
            output[:, x:min(420, x + width)] = centers[idx]
            x += width
        details = "Dominant RGB colours (approximate):\n" + "\n".join(
            f"RGB {tuple(int(v) for v in centers[idx])}: {counts[idx] / len(labels) * 100:.1f}%"
            for idx in order
        )
        return output, details
    return output, f"Image quantized to {k} dominant colours using K-means."



def freqfilter(image, lowpass=True):
    gray = togray(image).astype(np.float32)
    rows, cols = gray.shape
    dft = np.fft.fftshift(np.fft.fft2(gray))
    crow, ccol = rows // 2, cols // 2
    radius = max(5, min(rows, cols) // 8)
    mask = np.zeros((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    if not lowpass:
        mask = 1 - mask
    filtered = np.fft.ifft2(np.fft.ifftshift(dft * mask))
    magnitude = np.abs(filtered)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude, f"Applied a {'low-pass' if lowpass else 'high-pass'} frequency mask (radius {radius}px)."
def prcssimg(image, operation, settings, templateimg=None):
    rgb = torgb(image)
    gray = togray(rgb)
    t = int(settings.get("threshold", 127))
    t2 = int(settings.get("thresh2", 200))
    brightness = int(settings.get("brightness", 0))
    contrast = float(settings.get("contrast", 1.2))
    gamma = float(settings.get("gamma", 1.0))
    k = oddkrnl(settings.get("kernel", 5))
    minarea = int(settings.get("minarea", 40))
    invert = bool(settings.get("invert", True))
    hmin = int(settings.get("hmin", 0))
    hmax = int(settings.get("hmax", 179))
    smin = int(settings.get("smin", 40))
    vmin = int(settings.get("vmin", 40))
    strength = float(settings.get("strength", 1.0))
    if operation == "Brightness & contrast":
        out = cv2.convertScaleAbs(rgb, alpha=contrast, beta=brightness)
        return out, f"Brightness offset: {brightness}; contrast multiplier: {contrast:.2f}."
    if operation == "Gamma correction":
        lut = np.array([((i / 255.0) ** max(0.05, gamma)) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(rgb, lut), f"Gamma: {gamma:.2f}. Values below 1 brighten; above 1 darken."
    if operation == "Histogram equalization":
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), "Equalized the luminance channel while retaining colour information."
    if operation in ("CLAHE local contrast", "Medical contrast enhancement (CLAHE)"):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=max(1.0, strength * 2), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), f"CLAHE applied to luminance; clip limit: {max(1.0, strength * 2):.1f}."
    if operation == "Laplacian sharpening (class practical)":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        enhanced = cv2.subtract(gray, lap)
        return enhanced, "Laplacian response subtracted from the grayscale image, following the class practical approach."
    if operation == "Min-max normalization":
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX), "Grayscale intensity range stretched to 0–255."
    if operation == "Unsharp mask":
        blurred = cv2.GaussianBlur(rgb, (k, k), 0)
        return cv2.addWeighted(rgb, 1.6, blurred, -0.6, 0), f"Unsharp mask with Gaussian kernel {k} × {k}."
    if operation == "Average blur":
        return cv2.blur(rgb, (k, k)), f"Averaging kernel: {k} × {k}."
    if operation == "Gaussian blur":
        return cv2.GaussianBlur(rgb, (k, k), 0), f"Gaussian kernel: {k} × {k}."
    if operation == "Median filter":
        return cv2.medianBlur(rgb, k), f"Median kernel: {k} × {k}."
    if operation == "Medical denoising (median)":
        return cv2.medianBlur(rgb, k), f"Median denoising with kernel {k} × {k}; this is image processing, not diagnosis."
    if operation == "Minimum filter (class practical)":
        return cv2.erode(gray, np.ones((k, k), np.uint8)), f"Minimum filter implemented with grayscale erosion; kernel: {k} × {k}."
    if operation == "Maximum filter (class practical)":
        return cv2.dilate(gray, np.ones((k, k), np.uint8)), f"Maximum filter implemented with grayscale dilation; kernel: {k} × {k}."
    if operation == "Bilateral filter":
        return cv2.bilateralFilter(rgb, d=k, sigmaColor=75, sigmaSpace=75), f"Bilateral filter diameter: {k}; edges are better preserved than with a simple blur."
    if operation == "Sharpen filter":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(rgb, -1, kernel), "Applied a 3 × 3 sharpening kernel."
    if operation == "Emboss filter":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        embossed = cv2.filter2D(rgb, cv2.CV_16S, kernel) + 128
        return np.clip(embossed, 0, 255).astype(np.uint8), "Applied an emboss kernel."
    if operation in ("Sobel magnitude (class practical)", "Medical boundary extraction (Canny)"):
        if operation.startswith("Medical"):
            edges = cv2.Canny(gray, t, max(t + 1, t2))
            return edges, f"Canny boundaries: lower={t}, upper={max(t + 1, t2)}. This does not identify disease."
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(sx, sy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), f"Combined Sobel X and Y gradients; kernel size: {k}."
    if operation == "Sobel X":
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)), f"Horizontal Sobel derivative; kernel size: {k}."
    if operation == "Sobel Y":
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)), f"Vertical Sobel derivative; kernel size: {k}."
    if operation == "Prewitt edges":
        px = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32))
        py = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32))
        return cv2.convertScaleAbs(cv2.magnitude(px, py)), "Combined horizontal and vertical Prewitt responses."
    if operation == "Scharr edges":
        sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        return cv2.convertScaleAbs(cv2.magnitude(sx, sy)), "Combined Scharr X and Y responses."
    if operation == "Laplacian edges":
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F)), "Second-derivative Laplacian edge response."
    if operation == "Canny edges":
        return cv2.Canny(gray, t, max(t + 1, t2)), f"Canny thresholds: {t} and {max(t + 1, t2)}."
    if operation == "Roberts edges":
        gx = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, 0], [0, -1]], np.float32))
        gy = cv2.filter2D(gray, cv2.CV_32F, np.array([[0, 1], [-1, 0]], np.float32))
        return cv2.convertScaleAbs(cv2.magnitude(gx, gy)), "Roberts cross-gradient edge response."
    if operation == "Global threshold":
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        out = cv2.threshold(gray, t, 255, mode)[1]
        return out, f"Global threshold: {t}; {'foreground dark' if invert else 'foreground light'}."
    if operation in ("Otsu threshold", "Document binarization (Otsu)", "Medical region segmentation (Otsu)"):
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        actual_t, out = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
        return out, f"Otsu automatically selected threshold: {actual_t:.0f}."
    if operation == "Adaptive threshold":
        block = max(3, k if k % 2 else k + 1)
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY, block, 7)
        return out, f"Adaptive Gaussian threshold; block size: {block}; C=7."
    if operation == "HSV colour segmentation":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([min(hmin, hmax), smin, vmin], np.uint8)
        upper = np.array([max(hmin, hmax), 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_and(rgb, rgb, mask=mask), f"HSV range: H {lower[0]}–{upper[0]}, S ≥ {smin}, V ≥ {vmin}."
    if operation in ("Horizontal projection / line segmentation", "Line segmentation"):
        return projsegments(rgb, "horizontal", minarea)
    if operation in ("Vertical projection / word segmentation", "Word segmentation", "Character segmentation"):
        return projsegments(rgb, "vertical", minarea)
    if operation == "Connected-component segmentation":
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        labels, rows = regiontbl(binary, minarea)
        rng = np.random.default_rng(7)
        output = np.zeros((*gray.shape, 3), dtype=np.uint8)
        for row in rows:
            label = row["label"]
            output[labels == label] = rng.integers(60, 255, 3, dtype=np.uint8)
        return output, f"Connected components above min area {minarea}: {len(rows)}.\n" + "\n".join(str(r) for r in rows[:20])
    if operation == "Watershed segmentation":
        return watershed(rgb, minarea)
    if operation in ("Erosion", "Dilation", "Opening", "Closing", "Morphological gradient", "Top-hat", "Black-hat"):
        binary = makemask(gray, t, invert)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if operation == "Erosion": out = cv2.erode(binary, kernel, iterations=1)
        elif operation == "Dilation": out = cv2.dilate(binary, kernel, iterations=1)
        elif operation == "Opening": out = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == "Closing": out = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == "Morphological gradient": out = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        elif operation == "Top-hat": out = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
        else: out = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
        return out, f"{operation} with an elliptical {k} × {k} structuring element."
    if operation == "Draw external contours" or operation == "Contour drawing":
        out, contours, _ = contourovl(rgb, minarea, labels=True)
        return out, f"Drew {len(contours)} external contour(s); minimum area: {minarea}px²."
    if operation == "Contour bounding boxes":
        out, contours, _ = contourovl(rgb, minarea, boxes=True, labels=True)
        return out, f"Found {len(contours)} contour(s) and drew their bounding rectangles."
    if operation in ("Contour measurements", "Medical connected-region measurements"):
        out, contours, binary = contourovl(rgb, minarea, boxes=True, labels=True)
        lines = []
        for i, c in enumerate(contours[:30], 1):
            x, y, w, h = cv2.boundingRect(c)
            lines.append(f"Object {i}: area={cv2.contourArea(c):.1f}px², perimeter={cv2.arcLength(c, True):.1f}px, bbox=({x},{y},{w},{h})")
        return out, f"Measured {len(contours)} contour(s).\n" + ("\n".join(lines) or "No objects passed the minimum-area filter.")
    if operation in ("4-directional chain code (class practical)", "8-directional chain code (class practical)"):
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) >= minarea]
        if not contours:
            return rgb, "No contours found. Try a binary/high-contrast image."
        contour = max(contours, key=cv2.contourArea)
        connectivity = 4 if operation.startswith("4-") else 8
        codes = chaincode(contour, connectivity)
        out = rgb.copy()
        cv2.drawContours(out, [contour], -1, (40, 220, 80), 2)
        return out, f"Largest contour area: {cv2.contourArea(contour):.1f}px²\n{connectivity}-direction chain code length: {len(codes)}\nFirst 250 codes: {codes[:250]}"
    if operation in ("Object count", "Component counting"):
        _, contours, _ = contourovl(rgb, minarea)
        out, _, _ = contourovl(rgb, minarea, boxes=True, labels=True)
        return out, f"Estimated object count: {len(contours)} (external contours above {minarea}px²)."
    if operation == "Shape classification":
        return shapeclass(rgb, minarea)
    if operation == "Deskew document":
        return deskew(rgb)
    if operation == "OCR text extraction (Tesseract)":
        try:
            import pytesseract
            text = pytesseract.image_to_string(rgb, config="--psm 6")
            return rgb, "Extracted text (Tesseract OCR, page segmentation mode 6):\n\n" + (text.strip() or "No text was detected.")
        except Exception as exc:
            return rgb, "Tesseract OCR could not run. In Colab, run the setup cell that installs tesseract-ocr.\n" + str(exc)
    if operation == "Handwritten digit recognition (SVM)":
        return recogdigit(rgb, "SVM")
    if operation == "Handwritten digit recognition (KNN)":
        return recogdigit(rgb, "KNN")
    if operation in ("Line detection (Hough)", "Surface edge inspection"):
        return houghlines(rgb, t)
    if operation in ("ORB feature inspection", "ORB keypoints"):
        gray = togray(rgb)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        out = cv2.drawKeypoints(rgb, keypoints, None, color=(40, 220, 80), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return out, f"ORB keypoints detected: {len(keypoints)}. Descriptors: {'available' if descriptors is not None else 'none'}."
    if operation == "Template matching":
        return templatematch(rgb, templateimg, t)
    if operation == "Hu shape moments":
        _, contours, _ = contourovl(rgb, minarea)
        if not contours:
            return rgb, "No contour found for Hu moments."
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        log_moments = [-math.copysign(math.log10(abs(v) + 1e-30), v) for v in moments]
        out, _, _ = contourovl(rgb, minarea, boxes=True)
        return out, "Log-transformed Hu moments (shape descriptors):\n" + "\n".join(f"Hu {i+1}: {v:.6g}" for i, v in enumerate(log_moments))
    if operation == "HOG feature visualization":
        small = cv2.resize(gray, (max(64, gray.shape[1] // 2), max(64, gray.shape[0] // 2)))
        _, hog_image = feature.hog(small, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_image = exposure.rescale_intensity(hog_image, in_range="image", out_range=(0, 255)).astype(np.uint8)
        return hog_image, "Computed a Histogram of Oriented Gradients (HOG) visualization."
    if operation == "Pencil sketch":
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        dodge = cv2.divide(gray, 255 - blur, scale=256)
        return dodge, "Pencil-sketch effect using grayscale inversion and Gaussian dodge blending."
    if operation == "Canny line art":
        return cv2.Canny(gray, t, max(t + 1, t2)), "Black-and-white line art generated with Canny edges."
    if operation == "Blueprint threshold":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(3, k), 7)
        blue = np.zeros((*gray.shape, 3), dtype=np.uint8)
        blue[:, :, 0] = binary // 4
        blue[:, :, 1] = binary // 2
        blue[:, :, 2] = binary
        return blue, "Stylized blueprint-like threshold rendering (visual effect, not CAD vectorization)."
    if operation == "RGB channel — Red": return rgb[:, :, 0], "Red channel shown as grayscale intensity."
    if operation == "RGB channel — Green": return rgb[:, :, 1], "Green channel shown as grayscale intensity."
    if operation == "RGB channel — Blue": return rgb[:, :, 2], "Blue channel shown as grayscale intensity."
    if operation in ("HSV channel — Hue", "HSV channel — Saturation", "HSV channel — Value"):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        idx = {"HSV channel — Hue": 0, "HSV channel — Saturation": 1, "HSV channel — Value": 2}[operation]
        channel = hsv[:, :, idx]
        if idx == 0: channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        return channel, f"HSV {['Hue','Saturation','Value'][idx]} channel shown as grayscale."
    if operation == "Dominant-colour palette (K-means)":
        return domcolors(rgb, 5, False)
    if operation == "K-means colour quantization":
        return domcolors(rgb, 5, True)
    if operation == "Rotate 90° clockwise": return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE), "Rotated 90° clockwise."
    if operation == "Rotate 90° counter-clockwise": return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE), "Rotated 90° counter-clockwise."
    if operation == "Flip horizontally": return cv2.flip(rgb, 1), "Flipped horizontally."
    if operation == "Flip vertically": return cv2.flip(rgb, 0), "Flipped vertically."
    if operation == "Resize to 50%": return cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), "Resized to 50% of original dimensions."
    if operation == "Resize to 150%": return cv2.resize(rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), "Resized to 150% of original dimensions."
    if operation == "Fourier magnitude spectrum":
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))))
        return cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), "Displayed the centred logarithmic Fourier magnitude spectrum."
    if operation == "Fourier low-pass filter": return freqfilter(gray, True)
    if operation == "Fourier high-pass filter": return freqfilter(gray, False)
    raise ValueError(f"Operation not implemented: {operation}")

#
basepath = Path(__file__).resolve().parent
uploaddir = basepath / "uplds"
outdir = basepath / "outpts"
uploaddir.mkdir(exist_ok=True)
outdir.mkdir(exist_ok=True)
app = Flask(__name__)
app.secret_key = "locctestvarun12"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
def saveimg(image, folder):
    image = touint8(image)
    filename = "img-" + uuid.uuid4().hex[:12] + ".png"
    path = folder / filename
    if image.ndim == 2:
        Image.fromarray(image).save(path)
    else:
        Image.fromarray(torgb(image)).save(path)
    return filename
def loadimg(filename):
    if not filename:
        raise ValueError("Load an image first.")
    path = uploaddir / secure_filename(filename)
    if not path.exists():
        path = outdir / secure_filename(filename)
    if not path.exists():
        raise ValueError("The saved image could not be found. Please load it again.")
    image = Image.open(path).convert("RGB")
    return np.array(image)
def readsettings():
    return {
        "threshold": int(request.form.get("threshold", 100)),
        "thresh2": int(request.form.get("thresh2", 200)),
        "brightness": int(request.form.get("brightness", 0)),
        "contrast": float(request.form.get("contrast", 1.2)),
        "gamma": float(request.form.get("gamma", 1.0)),
        "kernel": int(request.form.get("kernel", 5)),
        "minarea": int(request.form.get("minarea", 40)),
        "invert": request.form.get("invert") == "yes",
        "strength": float(request.form.get("strength", 1.5)),
        "hmin": int(request.form.get("hmin", 0)),
        "hmax": int(request.form.get("hmax", 179)),
        "smin": int(request.form.get("smin", 40)),
        "vmin": int(request.form.get("vmin", 40)),
    }
def gettemplate():#Save optional tmplte img, if uplded
    file = request.files.get("templateimg")
    if file and file.filename:
        image = Image.open(file.stream).convert("RGB")
        filename = saveimg(np.array(image), uploaddir)
        return filename
    return session.get("templatefile")
PAGE = r"""<!doctype html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>

:root {
    color-scheme: dark;
    --bg: #090b10;
    --surface: #151821;
    --surface2: #202431;
    --primary: #64b5f6;
    --text: #e6e6ef;
    --muted: #999eaf;
    --border: #34394a;
}
* {
    box-sizing: border-box;
}
html {
    scroll-behavior: smooth;
}
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 15px/1.6 system-ui, sans-serif;
    animation: fadein .5s ease;
}
header {
    padding: 25px clamp(18px, 5vw, 72px);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}
.brand {
    font-weight: 850;
    font-size: 25px;
    letter-spacing: -1px;
}

.brand span {
    color: var(--primary);
}
header p {
    color: var(--muted);
    margin: 3px 0 0;
    font-size: 13px;
}
main {
    max-width: 1160px;
    margin: 35px auto;
    padding: 0 20px 50px;
}
.workflow {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: clamp(18px, 3vw, 30px);
    box-shadow: 0 12px 40px #0005;
    animation: rise .6s ease both;
}
h2 {
    font-size: 22px;
    margin: 0 0 22px;
}
.fieldgrid,
.compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}
.field {
    min-width: 0;
}
label {
    display: block;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}
input[type=file],
select {
    width: 100%;
    padding: 13px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface2);
    color: var(--text);
    font: inherit;
    transition: .2s;
}
input:focus,
select:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px #b8baff22;
}
.filebox {
    padding: 16px;
    border: 1px dashed #555b70;
    border-radius: 14px;
    background: var(--surface2);
    transition: .25s;
}

.filebox:hover {
    border-color: var(--primary);
}

.help,
.empty {
    color: var(--muted);
    font-size: 12px;
}

.settings {
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}

.settings summary {
    cursor: pointer;
    color: var(--primary);
    font-weight: 700;
    padding: 10px 0;
}

.paramgrid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
}

.param label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
}

.param output {
    color: var(--primary);
    font-weight: 700;
}

input[type=range] {
    width: 100%;
    accent-color: var(--primary);
    cursor: pointer;
}

.check {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-top: 18px;
}

.check label {
    margin: 0;
    font-weight: 500;
}

.check input {
    accent-color: var(--primary);
}

.template {
    margin-top: 15px;
    display: none;
}

.actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 25px;
}

button {
    padding: 14px 28px;
    border: 0;
    border-radius: 100px;
    background: var(--primary);
    color: #17172b;
    font: bold 14px system-ui;
    cursor: pointer;
    transition: .25s ease;
    box-shadow: 0 5px 20px #b8baff22;
}

button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px #b8baff40;
}

button:active {
    transform: scale(.96);
}

button:disabled {
    opacity: .5;
    cursor: wait;
}

.alert {
    white-space: pre-wrap;
    margin-top: 20px;
    padding: 14px;
    border-radius: 12px;
    background: #302438;
    border: 1px solid #654c75;
    color: #eedcff;
    font-size: 13px;
}

.results {
    margin-top: 28px;
    animation: rise .7s ease both;
}

.results h2 {
    margin-bottom: 16px;
}

.imagepane {
    min-width: 0;
    padding: 13px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    transition: .3s ease;
}

.imagepane:hover {
    transform: translateY(-5px);
    border-color: var(--primary);
    box-shadow: 0 12px 30px #0006;
}

.imagepane img {
    width: 100%;
    height: clamp(220px, 38vw, 480px);
    display: block;
    object-fit: contain;
    background: #08090d;
    border-radius: 12px;
}

.caption {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding: 12px 3px 2px;
    font-size: 13px;
    font-weight: 700;
}

.caption a {
    color: var(--primary);
    text-decoration: none;
}

.caption a:hover {
    text-decoration: underline;
}

/* Animations */

@keyframes fadein {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes rise {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Scrollbar */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg);
}

::-webkit-scrollbar-thumb {
    background: #454b60;
    border-radius: 20px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}
/* Mobile */
@media(max-width:760px) {
    .fieldgrid,
    .compare {
        grid-template-columns: 1fr;
    }
    .paramgrid {
        grid-template-columns: 1fr 1fr;
    }
    .actions button {
        width: 100%;
    }
}
@media(max-width:440px) {
    .paramgrid {
        grid-template-columns: 1fr;
    }
}
@media(prefers-reduced-motion:reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
}

/* =====================================================
   EASEL IMAGELAB V5 — DEEP SPACE / ELECTRIC BLUE
   CSS ATMOSPHERE ENGINE
   ===================================================== */

:root {
    color-scheme: dark;

    --bg: #02050c;
    --surface: rgba(5, 13, 29, 0.82);
    --surface2: rgba(9, 24, 48, 0.88);

    --primary: #168cff;
    --text: #e7f3ff;
    --muted: #7895b8;
    --border: rgba(40, 135, 255, 0.22);

    --blue-glow: #087cff;
    --cyan-glow: #35baff;
    --deep-blue: #041b3b;
}

/* GLOBAL FOUNDATION */

html {
    background: #02050c;
    scroll-behavior: smooth;
}

body {
    position: relative;
    isolation: isolate;
    overflow-x: hidden;

    background:
        radial-gradient(
            ellipse at 50% -20%,
            #092858 0%,
            transparent 55%
        ),
        radial-gradient(
            ellipse at 90% 60%,
            #041b3a 0%,
            transparent 45%
        ),
        #02050c;

    color: var(--text);
}

/* ANIMATED ATMOSPHERIC LIGHT */

body::before {
    content: "";
    position: fixed;
    inset: -50%;
    z-index: -2;
    pointer-events: none;

    background:
        radial-gradient(
            circle at 30% 40%,
            rgba(0, 110, 255, 0.17),
            transparent 28%
        ),
        radial-gradient(
            circle at 70% 30%,
            rgba(0, 75, 200, 0.16),
            transparent 30%
        ),
        radial-gradient(
            circle at 55% 75%,
            rgba(20, 145, 255, 0.11),
            transparent 32%
        );

    filter: blur(65px);

    animation: atmosphereMove 22s ease-in-out infinite alternate;
}

/* SUBTLE DIGITAL GRID */

body::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;

    background-image:
        linear-gradient(
            rgba(35, 130, 255, 0.045) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(35, 130, 255, 0.045) 1px,
            transparent 1px
        );

    background-size: 65px 65px;

    mask-image: linear-gradient(
        to bottom,
        black,
        transparent 85%
    );

    animation: gridDrift 35s linear infinite;
}

@keyframes atmosphereMove {
    0% {
        transform: translate3d(-3%, -2%, 0) scale(1);
    }
    50% {
        transform: translate3d(3%, 2%, 0) scale(1.08);
    }
    100% {
        transform: translate3d(-1%, 4%, 0) scale(1.13);
    }
}

@keyframes gridDrift {
    from { background-position: 0 0; }
    to { background-position: 65px 65px; }
}

/* HEADER — FUTURISTIC GLASS */

header {
    position: relative;
    z-index: 2;

    background: rgba(2, 8, 20, 0.72);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);

    border-bottom: 1px solid rgba(35, 130, 255, 0.25);

    box-shadow:
        0 10px 50px rgba(0, 75, 200, 0.09),
        0 1px 0 rgba(75, 170, 255, 0.08);

    animation: headerReveal 1s cubic-bezier(.16,1,.3,1) both;
}

@keyframes headerReveal {
    from {
        opacity: 0;
        transform: translateY(-25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.brand {
    letter-spacing: 3px;
    font-size: 27px;
    font-weight: 900;

    text-shadow:
        0 0 15px rgba(20, 130, 255, 0.4),
        0 0 35px rgba(20, 130, 255, 0.18);
}

.brand span {
    color: #42aaff;
    text-shadow:
        0 0 12px rgba(30, 145, 255, 0.9),
        0 0 35px rgba(0, 110, 255, 0.55);
}

/* MAIN WORKSPACE */

main {
    position: relative;
    z-index: 1;
    animation: workspaceReveal 1s .15s ease both;
}

@keyframes workspaceReveal {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.workflow {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            145deg,
            rgba(9, 27, 55, 0.91),
            rgba(3, 10, 23, 0.94)
        );

    border: 1px solid rgba(40, 140, 255, 0.24);
    border-radius: 24px;

    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);

    box-shadow:
        0 25px 90px rgba(0, 0, 0, 0.55),
        0 0 45px rgba(0, 100, 255, 0.09),
        inset 0 1px 0 rgba(130, 190, 255, 0.09);

    transition:
        border-color .4s ease,
        box-shadow .4s ease;
}

.workflow:hover {
    border-color: rgba(45, 150, 255, 0.48);

    box-shadow:
        0 25px 90px rgba(0, 0, 0, 0.6),
        0 0 55px rgba(0, 115, 255, 0.15);
}

.workflow > * {
    position: relative;
    z-index: 1;
}

/* HEADINGS */

h2 {
    color: #eaf5ff;
    letter-spacing: -.5px;
    text-shadow: 0 0 20px rgba(30, 130, 255, .2);
}

/* INPUTS AND SELECTORS */

input[type=file],
select {
    background: rgba(3, 13, 30, 0.9);
    color: #dceeff;

    border: 1px solid rgba(50, 130, 230, 0.3);
    border-radius: 13px;

    transition:
        border-color .3s ease,
        box-shadow .3s ease,
        background .3s ease,
        transform .3s ease;
}

input:hover,
select:hover {
    border-color: rgba(50, 155, 255, .65);
}

input:focus,
select:focus {
    outline: none;
    border-color: #168cff;

    box-shadow:
        0 0 0 3px rgba(0, 125, 255, .13),
        0 0 22px rgba(0, 125, 255, .19);
}

.filebox {
    background: rgba(5, 20, 42, .7);
    border: 1px dashed rgba(55, 145, 255, .45);

    transition: .35s ease;
}

.filebox:hover {
    background: rgba(8, 34, 70, .8);
    border-color: #168cff;

    box-shadow:
        0 0 25px rgba(0, 115, 255, .12),
        inset 0 0 25px rgba(0, 115, 255, .05);
}

/* SETTINGS PANEL */

.settings {
    border-top: 1px solid rgba(50, 130, 255, .2);
}

.settings summary {
    color: #45aaff;
    transition: .25s ease;
}

.settings summary:hover {
    color: #a4d9ff;
    text-shadow: 0 0 15px #087cff;
}

.param output {
    color: #45aaff;
    text-shadow: 0 0 12px rgba(0, 130, 255, .5);
}

input[type=range] {
    accent-color: #168cff;
    filter: drop-shadow(0 0 4px rgba(0, 125, 255, .45));
}

.check input {
    accent-color: #168cff;
}

/* PROCESS BUTTON — ELECTRIC BLUE */

button {
    position: relative;
    overflow: hidden;

    background: linear-gradient(
        110deg,
        #0755c9,
        #087ff5,
        #28adff
    );

    color: white;
    letter-spacing: .5px;

    box-shadow:
        0 0 18px rgba(0, 115, 255, .32),
        0 8px 30px rgba(0, 90, 255, .18);

    transition:
        transform .3s cubic-bezier(.2,.8,.2,1),
        box-shadow .3s ease,
        filter .3s ease;
}

button:hover {
    transform: translateY(-4px) scale(1.025);

    box-shadow:
        0 0 20px rgba(0, 145, 255, .65),
        0 0 55px rgba(0, 105, 255, .3);

    filter: brightness(1.17);
}

button:active {
    transform: scale(.96);
}

/* IMAGE PANELS */

.compare {
    perspective: 1200px;
}

.imagepane {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            145deg,
            rgba(9, 27, 53, .94),
            rgba(3, 9, 20, .98)
        );

    border: 1px solid rgba(40, 135, 255, .25);
    border-radius: 19px;
    box-shadow:
        0 15px 45px rgba(0, 0, 0, .35),
        inset 0 1px 0 rgba(120, 185, 255, .08);
    transition: all .45s cubic-bezier(.2,.8,.2,1);
}
.imagepane:hover {
    transform: translateY(-8px) rotateX(2deg);
    border-color: rgba(40, 155, 255, .75);
    box-shadow:
        0 25px 65px rgba(0, 0, 0, .55),
        0 0 35px rgba(0, 110, 255, .18);
}
.imagepane img {
    transition:
        transform .7s cubic-bezier(.2,.8,.2,1),
        filter .5s ease;
}
.imagepane:hover img {
    transform: scale(1.035);
    filter: brightness(1.08) contrast(1.04);
}
.caption {
    color: #c7e4ff;
}
.caption a {
    color: #42aaff;
    text-shadow: 0 0 10px rgba(0, 130, 255, .35);
}
.alert {
    background: rgba(0, 70, 150, .13);
    border: 1px solid rgba(30, 140, 255, .35);
    color: #b9ddff;
    box-shadow:
        inset 0 0 25px rgba(0, 110, 255, .05),
        0 0 20px rgba(0, 100, 255, .07);

    animation: resultReveal .6s ease both;
}
@keyframes resultReveal {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
/* SCROLLBAR */
::-webkit-scrollbar {
    width: 7px;
}
::-webkit-scrollbar-track {
    background: #02050c;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(
        180deg,
        #087fff,
        #06356f
    );
    border-radius: 20px;
    box-shadow: 0 0 12px rgba(0, 125, 255, .4);
}
/* ACCESSIBILITY */
@media(prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
        scroll-behavior: auto !important;
    }
}
</style>
</head>
<body>
<header><div class="brand">EASEL <span>ImageLab</span></div><p>Img processing workspace v4.15 by varun</p></header>
<main>
<form class="workflow" method="post" enctype="multipart/form-data" id="processForm">
<input type="hidden" name="action" value="process">
<h2>Upload a img you have & choose a processin method</h2>
<div class="fieldgrid">
  <div class="field">
    <label for="imagefile">Your image upload</label>
    <div class="filebox"><input id="imagefile" type="file" name="imagefile" accept="image/*">
    <p class="help">Select an image to upload. Maxcap20MB.</p></div>
  </div>
  <div class="field">
    <label for="workspace">Image work category</label>
    <select id="workspace" name="workspace">
      {% for name in workspaces.keys() %}<option value="{{ name }}" {% if name == selectedws %}selected{% endif %}>{{ name }}</option>{% endfor %}
    </select>
    <label for="operation" style="margin-top:14px">Processing/output to apply/use</label>
    <select id="operation" name="operation">
      {% for name in allowedops %}<option value="{{ name }}" {% if name == selectedop %}selected{% endif %}>{{ name }}</option>{% endfor %}
    </select>
  </div>
</div>
<details class="settings">
 <summary>Output settings ⚙️</summary>
 <p class="help">Adjust only what you need</p>
 <div class="paramgrid">
  <div class="param"><label>Threshold <output id="thval">100</output></label><input type="range" name="threshold" min="0" max="255" value="100" oninput="thval.value=this.value"></div>
  <div class="param"><label>Second threshold <output id="th2val">200</output></label><input type="range" name="thresh2" min="1" max="255" value="200" oninput="th2val.value=this.value"></div>
  <div class="param"><label>Kernel size <output id="kval">5</output></label><input type="range" name="kernel" min="1" max="31" step="2" value="5" oninput="kval.value=this.value"></div>
  <div class="param"><label>Brightness <output id="bval">0</output></label><input type="range" name="brightness" min="-100" max="100" value="0" oninput="bval.value=this.value"></div>
  <div class="param"><label>Contrast <output id="cval">1.2</output></label><input type="range" name="contrast" min=".2" max="3" step=".05" value="1.2" oninput="cval.value=this.value"></div>
  <div class="param"><label>Gamma <output id="gval">1.0</output></label><input type="range" name="gamma" min=".1" max="3" step=".05" value="1" oninput="gval.value=this.value"></div>
  <div class="param"><label>Minimum area <output id="mval">40</output></label><input type="range" name="minarea" min="0" max="5000" step="10" value="40" oninput="mval.value=this.value"></div>
  <div class="param"><label>Enhancement <output id="sval">1.5</output></label><input type="range" name="strength" min=".5" max="5" step=".1" value="1.5" oninput="sval.value=this.value"></div>
  <div class="param"><label>Hue minimum <output id="hminval">0</output></label><input type="range" name="hmin" min="0" max="179" value="0" oninput="hminval.value=this.value"></div>
  <div class="param"><label>Hue maximum <output id="hmaxval">179</output></label><input type="range" name="hmax" min="0" max="179" value="179" oninput="hmaxval.value=this.value"></div>
  <div class="param"><label>Saturation min <output id="sminval">40</output></label><input type="range" name="smin" min="0" max="255" value="40" oninput="sminval.value=this.value"></div>
  <div class="param"><label>Value minimum <output id="vminval">40</output></label><input type="range" name="vmin" min="0" max="255" value="40" oninput="vminval.value=this.value"></div>
 </div>
 <div class="check">
  <input type="checkbox" id="invert" name="invert" value="yes" checked>
  <label for="invert">Invert binary output (dark foreground)</label>
 </div>
 <div class="template field" id="templateBox">
  <label for="templateimg">Template image (template matching only)</label>
  <input id="templateimg" type="file" name="templateimg" accept="image/*">
 </div>
</details>
<div class="actions">
 <button id="processBtn" type="submit">Process Now!</button>
</div>
{% if details %}
<div class="alert">{{ details }}</div>
{% endif %}
</form>
{% if originalfile and processedfile %}
<section class="results">
<h2>Processed result</h2>
<div class="compare">
 <div class="imagepane"><img src="{{ url_for('showimg', filename=originalfile) }}" alt="Uploaded original"><div class="caption">Uploaded image</div></div>
 <div class="imagepane"><img src="{{ url_for('showimg', filename=processedfile) }}" alt="Processed image"><div class="caption">Processed image <a href="{{ url_for('downloadimg', filename=processedfile) }}">Download</a></div></div>
</div>
</section>
{% elif originalfile %}
<section class="results"><p class="empty">Image uploaded. Choose the category and output above, then click Process image.</p></section>
{% endif %}
</main>
<script>
const workspaces={{ workspaces|tojson }};
const allops={{ allops|tojson }};
const ws=document.getElementById('workspace'), op=document.getElementById('operation');
const templateBox=document.getElementById('templateBox');
function refreshOps(){
 const previous=op.value, choices=(workspaces[ws.value]||[]).length?workspaces[ws.value]:allops;
 op.innerHTML='';
 choices.forEach(x=>{const option=document.createElement('option');option.value=x;option.textContent=x;op.appendChild(option)});
 if(choices.includes(previous))op.value=previous;
 refreshTemplate();
}
function refreshTemplate(){templateBox.style.display=op.value==='Template matching'?'block':'none'}
ws.addEventListener('change',refreshOps);op.addEventListener('change',refreshTemplate);refreshTemplate();
document.getElementById('processForm').addEventListener('submit',()=>{
 const btn=document.getElementById('processBtn');btn.disabled=true;btn.textContent='Processing…';
});
</script>
</body></html>"""
def showpage():
    selectedws = session.get("workspace", "Universal Lab")
    selectedop = session.get("operation", "Canny edges")
    allowedops = allops if selectedws == "Universal Lab" else workspaces.get(selectedws, allops)
    if not allowedops:
        allowedops = allops
    if selectedop not in allowedops:
        selectedop = allowedops[0]
        session["operation"] = selectedop
    return render_template_string(
        PAGE,
        workspaces=workspaces,
        allops=allops,
        allowedops=allowedops,
        selectedws=selectedws,
        selectedop=selectedop,
        originalfile=session.get("originalfile"),
        processedfile=session.get("processedfile"),
        details=session.get("details", ""),
    )
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return showpage()
    if request.form.get("action") != "process":
        return showpage()
    session["workspace"] = request.form.get("workspace", "Universal Lab")
    session["operation"] = request.form.get("operation", "Canny edges")
    uploaded = request.files.get("imagefile")
    try:
        if uploaded and uploaded.filename:
            image = Image.open(uploaded.stream).convert("RGB")
            original = np.array(image)
            originalfile = saveimg(original, uploaddir)
            session["originalfile"] = originalfile
        elif session.get("originalfile"):
            original = loadimg(session["originalfile"])
        else:
            session["details"] = "Choose an image before processing."
            return showpage()
        settings = readsettings()
        templatefile = gettemplate()
        templateimg = loadimg(templatefile) if templatefile else None
        result, details = prcssimg(original, session["operation"], settings, templateimg)
        processedfile = saveimg(result, outdir)
        session["processedfile"] = processedfile
        session["details"] = "Done — " + session["operation"] + ". " + details
    except Exception as exc:
        session["details"] = "Processing failed: " + type(exc).__name__ + ": " + str(exc)
    return showpage()
@app.route("/image/<filename>")
def showimg(filename):
    filename = secure_filename(filename)
    if (uploaddir / filename).exists():
        return send_from_directory(uploaddir, filename)
    if (outdir / filename).exists():
        return send_from_directory(outdir, filename)
    return "Image not found", 404
@app.route("/download/<filename>")
def downloadimg(filename):
    filename = secure_filename(filename)
    if (outdir / filename).exists():
        return send_from_directory(outdir, filename, as_attachment=True)
    if (uploaddir / filename).exists():
        return send_from_directory(uploaddir, filename, as_attachment=True)
    return "Image not found", 404
if __name__ == "__main__":
    print("Starting.......")
    print("address to go in ur browser- http://127.0.0.1:5000")
    app.run(debug=True)