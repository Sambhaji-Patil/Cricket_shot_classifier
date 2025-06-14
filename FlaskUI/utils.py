def process_images(img_path, output_folder):
    from predict_image import predict_image  # use your final version
    from gradcam import grad_cam_visualization  # use your fixed Grad-CAM code
    import os
    import shutil

    label, probs = predict_image(img_path)

    gradcam_path = os.path.join(output_folder, f"gradcam_{os.path.basename(img_path)}")
    grad_cam_visualization(img_path, gradcam_path)

    return {
        'filename': os.path.basename(img_path),
        'filepath': img_path,
        'gradcam': gradcam_path,
        'predicted': label,
        'probabilities': probs
    }