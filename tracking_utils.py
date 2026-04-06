import cv2
import numpy as np
import time

def track_and_project(system, image_to_project, printed_aruco_id=23):
    """
    Projects an image at the original corner placement and keeps it at the same
    position relative to the printed ArUco marker using tracking.
    
    Uses the same method as realtime_tracking_v3 (screen-space offset approach).
    
    Args:
        system: CaptureSystem object with camera, projector, and ArUco setup
        image_to_project: Image (numpy array) to project
        printed_aruco_id: ID of the printed ArUco marker (default: 23)
    
    Returns:
        None (runs until 'q' is pressed)
    """
    # Setup ArUco detector
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(system.aruco_dict, detectorParams)
    
    def get_marker_corners(corners, ids, target_id):
        if ids is None:
            return None
        idx = np.where(ids.flatten() == target_id)[0]
        if len(idx) == 0:
            return None
        return corners[idx[0]].reshape(4, 2)
    
    # Create windows
    cv2.namedWindow("Projection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("Tracking Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking Preview", 800, 600)
    
    # Get original screen corners from system
    original_screen_corners = system.orig_proj_corners.astype(np.float32)
    
    # Prepare the image to project
    if len(image_to_project.shape) == 2:
        img_to_proj = cv2.cvtColor(image_to_project, cv2.COLOR_GRAY2BGR)
    else:
        img_to_proj = image_to_project.copy()
    
    # Source corners for perspective transform
    src_corners = np.array([
        [0, 0],
        [img_to_proj.shape[1], 0],
        [img_to_proj.shape[1], img_to_proj.shape[0]],
        [0, img_to_proj.shape[0]]
    ], dtype=np.float32)
    
    # Helper function to render projection
    def render_projection(screen_corners):
        M = cv2.getPerspectiveTransform(src_corners, screen_corners.astype(np.float32))
        proj_img = cv2.warpPerspective(
            img_to_proj, M, 
            (system.screen_res[0], system.screen_res[1])
        )
        if len(system.img.shape) == 2:
            proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Projection", proj_img)
    
    # Show initial projection at original corners
    render_projection(original_screen_corners)
    
    print("=== TRACK AND PROJECT ===")
    print("Waiting for calibration... Detecting printed marker...")
    print("Press 'q' to quit, 'r' to reset calibration")
    
    # Calibration phase: detect printed marker and capture current projection position
    original_printed_corners = None
    
    while original_printed_corners is None:
        ret, frame = system.cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        original_printed_corners = get_marker_corners(corners, ids, printed_aruco_id)
        
        frame_viz = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_viz, corners, ids)
        
        status = "Calibrating... "
        if original_printed_corners is None:
            status += f"[Waiting for printed #{printed_aruco_id}]"
        else:
            status += "[READY - Calibrating...]"
        
        cv2.putText(frame_viz, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Tracking Preview", frame_viz)
        render_projection(original_screen_corners)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return
    
    print("Calibration complete!")
    
    # Use system's pre-calibrated projection corners
    if hasattr(system, 'corners_img_proj') and system.corners_img_proj is not None:
        original_projected_corners = system.corners_img_proj.astype(np.float32)
    else:
        print("Warning: No pre-calibrated projection corners found.")
        print("Using default assumption based on system geometry.")
        original_projected_corners = original_screen_corners.copy()
    
    # Compute screen-space offset (same as V3)
    H_screen_to_camera_orig = cv2.findHomography(
        original_screen_corners,
        original_projected_corners
    )[0]
    H_camera_to_screen_orig = np.linalg.inv(H_screen_to_camera_orig)
    
    # Where printed marker is in screen space at calibration
    printed_in_screen_orig = cv2.perspectiveTransform(
        original_printed_corners.reshape(1, -1, 2).astype(np.float32),
        H_camera_to_screen_orig
    ).reshape(-1, 2)
    
    # Screen-space offset
    screen_offset = original_screen_corners - printed_in_screen_orig
    
    print(f"Screen offset computed:\n{screen_offset}")
    print(f"Printed marker in screen space (orig):\n{printed_in_screen_orig}")
    print("\nTracking started! Move the printed marker to see projection follow.")
    
    # Tracking loop
    fps_start = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = system.cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        current_printed = get_marker_corners(corners, ids, printed_aruco_id)
        
        frame_viz = frame.copy()
        
        if current_printed is not None:
            # Draw printed marker (green)
            pts = current_printed.astype(int)
            cv2.polylines(frame_viz, [pts], True, (0, 255, 0), 2)
            
            # Map current printed marker position to screen space
            printed_in_screen_curr = cv2.perspectiveTransform(
                current_printed.reshape(1, -1, 2).astype(np.float32),
                H_camera_to_screen_orig
            ).reshape(-1, 2)
            
            # Apply screen-space offset to get new screen corners
            new_screen_corners = printed_in_screen_curr + screen_offset
            
            # Update projection with new corners
            render_projection(new_screen_corners)
            
            # Compute where projection should appear in camera (for visualization)
            desired_proj_corners = cv2.perspectiveTransform(
                new_screen_corners.reshape(1, -1, 2).astype(np.float32),
                H_screen_to_camera_orig
            ).reshape(-1, 2)
            
            # Draw desired position (yellow)
            pts_desired = desired_proj_corners.astype(int)
            cv2.polylines(frame_viz, [pts_desired], True, (0, 255, 255), 2)
            
            status = "TRACKING"
            status_color = (0, 255, 0)
        else:
            status = "LOST - Looking for printed marker..."
            status_color = (0, 0, 255)
            # Keep showing last projection
        
        # FPS
        frame_count += 1
        if time.time() - fps_start >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_start = time.time()
        
        cv2.putText(frame_viz, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame_viz, f"FPS: {fps}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_viz, "Press 'q' to quit, 'r' to reset", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Tracking Preview", frame_viz)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset calibration
            if current_printed is not None:
                original_printed_corners = current_printed.copy()
                
                # Recapture projection corners
                if hasattr(system, 'corners_img_proj') and system.corners_img_proj is not None:
                    original_projected_corners = system.corners_img_proj.astype(np.float32)
                else:
                    original_projected_corners = original_screen_corners.copy()
                
                H_screen_to_camera_orig = cv2.findHomography(
                    original_screen_corners,
                    original_projected_corners
                )[0]
                H_camera_to_screen_orig = np.linalg.inv(H_screen_to_camera_orig)
                
                printed_in_screen_orig = cv2.perspectiveTransform(
                    original_printed_corners.reshape(1, -1, 2).astype(np.float32),
                    H_camera_to_screen_orig
                ).reshape(-1, 2)
                
                screen_offset = original_screen_corners - printed_in_screen_orig
                print("Calibration reset!")
                print(f"New screen offset:\n{screen_offset}")
    
    cv2.destroyAllWindows()
    print("Tracking ended.")

print("track_and_project() function defined!")
print("\nUsage: track_and_project(system, your_image)")
print("  - system: CaptureSystem object")
print("  - your_image: numpy array of image to project")


import cv2
import numpy as np
import time
import torch


class TrackerSystem:
    def __init__(self, system, predict_raw, weights, printed_aruco_id=10):
        """
        Initialize the tracking and classification system.
        
        Args:
            system: CaptureSystem object with camera, projector, and ArUco setup
            predict_raw: Classification function (takes batched tensor, returns predictions)
            weights: Model weights with meta["categories"] for class labels
            printed_aruco_id: ID of the printed ArUco marker (default: 10)
        """
        self.system = system
        self.predict_raw = predict_raw
        self.weights = weights
        self.printed_aruco_id = printed_aruco_id
        
        # Setup ArUco detector
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(system.aruco_dict, self.detectorParams)
        
        # Create windows
        cv2.namedWindow("Projection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Projection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Classification", 800, 600)
        
        # Get original screen corners from system
        self.original_screen_corners = system.orig_proj_corners.astype(np.float32)
        
        # Tensor conversion helper
        self.tt = lambda x: torch.tensor(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255.).permute(2,0,1).float()
        
        print("TrackerSystem initialized!")
    
    def get_marker_corners(self, corners, ids, target_id):
        """Helper to extract marker corners by ID."""
        if ids is None:
            return None
        idx = np.where(ids.flatten() == target_id)[0]
        if len(idx) == 0:
            return None
        return corners[idx[0]].reshape(4, 2)
    
    def track_project_and_classify(self, image_to_project):
        """
        Projects an image with tracking while performing classification and displaying
        prediction results with probability.
        
        Similar to track_and_project but:
        - No visual markers/outlines on preview
        - Shows classification prediction and probability on captured frame
        - Collects results for later analysis
        
        Args:
            image_to_project: Image (numpy array) to project
        
        Returns:
            tuple: (caps, results) - list of captured frames and classification results
        """
        # Prepare the image to project
        if len(image_to_project.shape) == 2:
            img_to_proj = cv2.cvtColor(image_to_project, cv2.COLOR_GRAY2BGR)
        else:
            img_to_proj = image_to_project.copy()
        
        # Source corners for perspective transform
        src_corners = np.array([
            [0, 0],
            [img_to_proj.shape[1], 0],
            [img_to_proj.shape[1], img_to_proj.shape[0]],
            [0, img_to_proj.shape[0]]
        ], dtype=np.float32)
        
        # Helper function to render projection
        def render_projection(screen_corners):
            M = cv2.getPerspectiveTransform(src_corners, screen_corners.astype(np.float32))
            proj_img = cv2.warpPerspective(
                img_to_proj, M, 
                (self.system.screen_res[0], self.system.screen_res[1])
            )
            cv2.imshow("Projection", proj_img)
        
        # Show initial projection at original corners
        render_projection(self.original_screen_corners)
        
        print("=== TRACK, PROJECT, AND CLASSIFY ===")
        print("Waiting for calibration... Detecting printed marker...")
        print("Press 'q' to quit, 'r' to reset calibration")
        
        # Calibration phase: detect printed marker
        original_printed_corners = None
        
        while original_printed_corners is None:
            ret, frame = self.system.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            original_printed_corners = self.get_marker_corners(corners, ids, self.printed_aruco_id)
            
            frame_viz = frame.copy()
            
            status = "Calibrating... "
            if original_printed_corners is None:
                status += f"[Waiting for printed #{self.printed_aruco_id}]"
            else:
                status += "[READY - Press any key to start]"
            
            cv2.putText(frame_viz, status, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Classification", frame_viz)
            render_projection(self.original_screen_corners)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return [], []
        
        print("Calibration complete! Starting classification...")
        
        # Use system's pre-calibrated projection corners
        if hasattr(self.system, 'corners_img_proj') and self.system.corners_img_proj is not None:
            original_projected_corners = self.system.corners_img_proj.astype(np.float32)
        else:
            print("Warning: No pre-calibrated projection corners found.")
            original_projected_corners = self.original_screen_corners.copy()
        
        # Compute screen-space offset
        H_screen_to_camera_orig = cv2.findHomography(
            self.original_screen_corners,
            original_projected_corners
        )[0]
        H_camera_to_screen_orig = np.linalg.inv(H_screen_to_camera_orig)
        
        # Where printed marker is in screen space at calibration
        printed_in_screen_orig = cv2.perspectiveTransform(
            original_printed_corners.reshape(1, -1, 2).astype(np.float32),
            H_camera_to_screen_orig
        ).reshape(-1, 2)
        
        # Screen-space offset
        screen_offset = self.original_screen_corners - printed_in_screen_orig
        
        # Storage for results
        caps = []
        results = []
        
        # Tracking and classification loop
        fps_start = time.time()
        frame_count = 0
        fps = 0
        
        print("Tracking and classifying... Press 'q' to stop")
        
        while True:
            ret, frame = self.system.cap.read()
            frame_display = frame.copy()

            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            current_printed = self.get_marker_corners(corners, ids, self.printed_aruco_id)
            
            # Update projection if marker detected
            if current_printed is not None:
                # Map current printed marker position to screen space
                printed_in_screen_curr = cv2.perspectiveTransform(
                    current_printed.reshape(1, -1, 2).astype(np.float32),
                    H_camera_to_screen_orig
                ).reshape(-1, 2)
                
                # Apply screen-space offset to get new screen corners
                new_screen_corners = printed_in_screen_curr + screen_offset
                
                # add corners to frame display for visualization
                pts = current_printed.astype(int)
                cv2.polylines(frame_display, [pts], True, (255, 0, 0), 2)

                # Update projection
                render_projection(new_screen_corners)
            else:
                # Indicate lost marker at the bottom of the frame
                cv2.putText(frame_display, "Marker LOST", (10, frame_display.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                

            
                
            
            # Perform classification
            tr = self.tt(frame)
            with torch.no_grad():
                p = self.predict_raw(tr.unsqueeze(0).cuda())
                res = self.weights.meta["categories"][p[0].argmax(0).item()]
                prob = p[0].max(0).values.item() * 100
            
            # Create display frame with prediction text
            cv2.putText(frame_display, f'Pred: {res}: {prob:.2f}%', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # FPS counter
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_start = time.time()
            
            cv2.putText(frame_display, f'FPS: {fps}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Classification", frame_display)
            
            # Store results
            results.append(res)
            caps.append(frame_display.copy())
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset calibration
                if current_printed is not None:
                    original_printed_corners = current_printed.copy()
                    
                    if hasattr(self.system, 'corners_img_proj') and self.system.corners_img_proj is not None:
                        original_projected_corners = self.system.corners_img_proj.astype(np.float32)
                    else:
                        original_projected_corners = self.original_screen_corners.copy()
                    
                    H_screen_to_camera_orig = cv2.findHomography(
                        self.original_screen_corners,
                        original_projected_corners
                    )[0]
                    H_camera_to_screen_orig = np.linalg.inv(H_screen_to_camera_orig)
                    
                    printed_in_screen_orig = cv2.perspectiveTransform(
                        original_printed_corners.reshape(1, -1, 2).astype(np.float32),
                        H_camera_to_screen_orig
                    ).reshape(-1, 2)
                    
                    screen_offset = self.original_screen_corners - printed_in_screen_orig
                    print("Calibration reset!")
        
        cv2.destroyAllWindows()
        print(f"Classification ended. Captured {len(caps)} frames.")
        
        return caps, results


print("TrackerSystem class defined!")
print("\nUsage:")
print("  from classfier import *")
print("  tracker = TrackerSystem(system, predict_raw, weights, printed_aruco_id=10)")
print("  caps, results = tracker.track_project_and_classify(image_to_project)")
print("\nReturns: (caps, results) - captured frames and classification results")