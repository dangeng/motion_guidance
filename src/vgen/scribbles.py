import gradio as gr
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch.backends
import torch.backends.mps

# SAM model
SAM_CHECKPOINT_PATH = "./assets/sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)

with gr.Blocks() as demo:
    # --------------------------------------------------------------------
    # --                     Gradio Components                          --
    # --------------------------------------------------------------------
    
    # Images
    image_input = gr.Image(type="numpy", label="Upload an Image", interactive=True, visible=True)
    image_editing = gr.Paint(label="Draw Mask", type="numpy", interactive=False, layers=False)
    image_guidance = gr.Image(interactive=False)

    # States
    state_translation_center = gr.State()
    state_motion_option = gr.State(value = "Segment")
    
    # Textboxes
    text_translation_center = gr.Textbox(placeholder="Click on the image to get the center of the translation", interactive=False)
    
    # Buttons
    button_movement = gr.Radio(["Segment", "Translate"], label="Motion Guidance", info="Choose among the available motion guidance options", value="Segment", visible=False, interactive=False)
    
    # --------------------------------------------------------------------
    # --                     Gradio Events                             --
    # --------------------------------------------------------------------
    # Upload image
    @image_input.upload(
        inputs=image_input,
        outputs=[image_input, image_editing, button_movement],
    )
    def on_upload(image):
        """Hides the static uploaded image and shows the editable image"""
        return gr.update(visible=False), image, gr.update(visible=True)
    
    # Click on image
    @image_editing.select(
        inputs=image_input,
        outputs=[image_guidance, 
                 image_editing, 
                 state_translation_center,
                 button_movement],
    )
    def on_image_click(image, evt: gr.SelectData):
        """Get clicked coordinates and run SAM model"""

        clicked_points = evt.index
        
        # Run SAM model
        predictor.set_image(image)
        input_point = np.array([clicked_points])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask = masks[scores.argmax()]
        np.save('mask.npy', mask)
        
        # Overlay mask on image
        mask = mask.astype(np.uint8) * 255 
        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = mask 
        
        # Find center of color mask
        mask_center = np.mean(np.argwhere(mask > 0), axis=0).astype(int)[::-1]
        
        # Overlay the mask and draw a circle at the center
        masked_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
        masked_image = cv2.circle(masked_image, tuple(mask_center), 5, (0, 0, 255), -1)
        
        # Update the image, and save the mask center
        return masked_image, masked_image, mask_center, gr.update(interactive=True)

    # Update text box with the center of the translation
    @state_translation_center.change(
        inputs=state_translation_center,
        outputs=text_translation_center,
    )
    def on_state_translation_center_change(state_translation_center):
        """Update text box with the center of the translation"""
        if state_translation_center is not None:
            return f"Center of translation: {state_translation_center[0]}, {state_translation_center[1]}"
        else:
            return "Click on the image to get the center of the translation"
    
    @image_editing.apply(
        inputs=[image_editing, state_translation_center, state_motion_option],
        outputs=image_guidance,
    )
    def on_paint(image, mask_center, motion_state):
        """Apply an approximation of the motion from the mask center
        to the scribble."""
        
        # If no mask center is provided, return the original image
        if mask_center is None:
            return image['background']
        
        # Make the mask binary
        mask = cv2.cvtColor(image['layers'][0], cv2.COLOR_RGBA2GRAY)
        mask[mask > 0] = 1
        
        # If no mask is drawn, return the original image
        if mask.sum() == 0:
            return image['background']
        
        if motion_state == "Translate":
            # Extract the minimum and maximum coordinates of the mask
            y_coords, x_coords = np.where(mask > 0)
            coords = np.array([x_coords, y_coords]).T
            
            # See which if the coordinate is closer to the mask center
            distances = np.linalg.norm(mask_center - coords, axis=1)
            start_point = mask_center
            end_point = coords[np.argmax(distances)]
            
            # Draw arrow from the center of the mask to the clicked point
            arrow_color = (0, 255, 0)
            arrow_width = 5
            image = cv2.arrowedLine(img = image['background'],
                                    pt1 = tuple(start_point), 
                                    pt2 = tuple(end_point),
                                    color = arrow_color, 
                                    thickness=arrow_width, 
                                    )
            return image
        else:
            raise NotImplementedError("Only translation is implemented for now.")
    
    # Make the editing interactable depending on the button state
    @button_movement.select(
        inputs=[],
        outputs=[image_editing, state_motion_option],
    )
    def on_button_select(evt: gr.SelectData):
        """Change the button state and update the stored state"""
        
        if evt.value == "Segment":
            return gr.update(interactive=False), evt.value
        elif evt.value == "Translate":
            return gr.update(interactive=True), evt.value
        else:
            raise ValueError("When did this happen?")
        
    
if __name__ == "__main__":  
    demo.launch()
