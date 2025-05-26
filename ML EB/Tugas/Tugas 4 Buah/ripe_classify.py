import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import os

FEATURE_NAMES = [
    "Mean H", "Mean S", "Mean V",
    "Std H", "Std S", "Std V",
    "Ratio S/H", "Ratio V/S",
    "Entropy H",
    "Prop Kuning", "Prop Hijau",
    "Std Gray"
]

def extract_extended_features(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    ratio_s_h = mean_s / (mean_h + 1e-5)
    ratio_v_s = mean_v / (mean_s + 1e-5)

    hist_h = cv2.calcHist([hsv], [0], None, [180], [0,180])
    hist_h = hist_h / hist_h.sum()
    entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))

    total_pixels = h.size
    prop_kuning = np.sum((h >= 20) & (h <= 40)) / total_pixels
    prop_hijau  = np.sum((h >= 50) & (h <= 70)) / total_pixels

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_gray = np.std(gray)

    return np.array([
        mean_h, mean_s, mean_v,
        std_h, std_s, std_v,
        ratio_s_h, ratio_v_s,
        entropy_h,
        prop_kuning, prop_hijau,
        std_gray
    ])

def train_classifier(features_df, model_type='rf'):
    if 'Label' in features_df.columns and 'Filename' in features_df.columns:
        X = features_df.drop(['Label', 'Filename'], axis=1)
        y = features_df['Label']
    else:
        X = features_df.iloc[:, :-2]
        y = features_df.iloc[:, -2]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        clf = SVC(probability=True, random_state=42)
    elif model_type == 'optimal':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        clf = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    
    clf.fit(X_train, y_train)
    
    if model_type == 'optimal':
        print(f"Best parameters: {clf.best_params_}")
        clf = clf.best_estimator_
    
    accuracy = clf.score(X_test, y_test)
    print(f"Model {model_type} accuracy: {accuracy:.4f}")
    
    if hasattr(clf, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(feature_importance.head(5))
    
    with open(f'fruit_classifier_{model_type}.pkl', 'wb') as f:
        pickle.dump((clf, scaler, X.columns.tolist()), f)
    
    return clf, scaler, X.columns.tolist()

def real_time_classification(model_type='rf'):
    model_path = f'fruit_classifier_{model_type}.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            try:
                clf, scaler, feature_names = pickle.load(f)
                print(f"Model {model_type} loaded successfully")
            except ValueError:
                f.seek(0)
                clf, scaler = pickle.load(f)
                feature_names = FEATURE_NAMES
                
                with open(model_path, 'wb') as f_new:
                    pickle.dump((clf, scaler, feature_names), f_new)
    except Exception as e:
        print(f"Loading model error: {e}")
        print("Training new model...")
        all_features = pd.read_csv('fitur_buah.csv')
        clf, scaler, feature_names = train_classifier(all_features, model_type)
    
    for cam_index in [0, 1, 2]:
        print(f"Testing camera {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Camera {cam_index} connected")
            break
        cap.release()
    else:
        print("No camera found")
        return
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution: {width}x{height}")
    
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame grab failed #{frame_counter}")
            if frame_counter > 0:
                cap.release()
                cap = cv2.VideoCapture(cam_index)
                if not cap.isOpened():
                    print("Camera reconnection failed")
                    break
                continue
            else:
                break
            
        frame_counter += 1
        
        cv2.imshow("Camera Feed", frame)
        
        h, w, _ = frame.shape
        crop_w = int(w * 0.25)
        crop_h = int(h * 0.25)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
            continue
            
        roi = frame[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            continue
            
        cv2.imshow("ROI", roi)
        
        try:
            features = extract_extended_features(roi)
            features_df = pd.DataFrame([features], columns=feature_names)
            features_scaled = scaler.transform(features_df)
            
            prediction = clf.predict(features_scaled)[0]
            probabilities = clf.predict_proba(features_scaled)[0]
            confidence = max(probabilities) * 100
            
            if prediction == 'matang':
                color = (0, 255, 255)
            elif prediction == 'mentah':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            cv2.putText(overlay, f"Prediction: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(overlay, f"Confidence: {confidence:.2f}%", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            y_pos = 110
            key_features = ['Prop Kuning', 'Prop Hijau', 'Ratio S/H', 'Ratio V/S']
            for i, name in enumerate(key_features):
                try:
                    idx = feature_names.index(name)
                    value = features[idx]
                except ValueError:
                    idx = {'Prop Kuning': 9, 'Prop Hijau': 10, 'Ratio S/H': 6, 'Ratio V/S': 7}.get(name, 0)
                    value = features[idx]
                
                cv2.putText(overlay, f"{name}: {value:.2f}", (10, y_pos + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Fruit Ripeness Classification", overlay)
            
        except Exception as e:
            print(f"Processing error: {e}")
            cv2.imshow("Fruit Ripeness Classification", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        all_features = pd.read_csv('fitur_buah.csv')
        print(f"Data: {all_features.shape} rows")
    except Exception as e:
        print(f"Data loading error: {e}")
        exit(1)
    
    model_type = 'optimal' 
    
    model_path = f'fruit_classifier_{model_type}.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                if len(model_data) != 3:
                    print("Updating model format...")
                    os.remove(model_path)
        except:
            print("Recreating corrupted model...")
            os.remove(model_path)
    
    if not os.path.exists(model_path):
        print(f"Training {model_type} model...")
        train_classifier(all_features, model_type)
    
    real_time_classification(model_type)