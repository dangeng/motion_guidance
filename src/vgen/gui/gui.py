import torch
import numpy as np
import math
import pygame
from PIL import Image

MODES = ['translate', 'rotate', 'scale', 'scale_1d']
MODE = MODES[0]
im_path = './assets/apple.png'
im_path = './assets/topiary.png'
im = Image.open(im_path)
im = np.array(im)
im = im[...,:-1] # Remove alpha channel
shade = 0.5

### START SAM STUFF ###
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = "./assets/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(im)
### END SAM STUFF ###

if MODE == 'translate':
    from vgen.gui.get_image import get_translation as get_image
elif MODE == 'rotate':
    from vgen.gui.get_image import get_rotation as get_image
elif MODE == 'scale':
    from vgen.gui.get_image import get_scale as get_image
elif MODE == 'scale_1d':
    from vgen.gui.get_image import get_scale_1d as get_image

def show_image(image):
    image_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    screen.blit(image_surface, (0, 0))
    pygame.display.flip()

# Initialize Pygame
pygame.init()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Set up the screen
screen_width, screen_height = 512, 512
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Arrow App")
screen.fill(white)

# Draw image
image = Image.open(im_path)
show_image(image)

# Arrow properties
arrow_start = None
arrow_end = None
drawing = False

def draw_arrow(surface, color, start, end, width):
    arrow_size = 7
    pygame.draw.line(surface, color, start, end, width)
    theta = math.atan2((start[0]-end[0]), (start[1]-end[1])) + math.pi / 3
    pygame.draw.polygon(surface, color, (
                  (end[0]+arrow_size*math.sin(theta), 
                   end[1]+arrow_size*math.cos(theta)), 
                  (end[0]+arrow_size*math.sin(theta+2*math.pi/3), 
                   end[1]+arrow_size*math.cos(theta+2*math.pi/3)), 
                  (end[0]+arrow_size*math.sin(theta-2*math.pi/3), 
                   end[1]+arrow_size*math.cos(theta-2*math.pi/3))))

# Main game loop
running = True
getting_mask = True
while running:
    for event in pygame.event.get():
        if getting_mask:
            if event.type == pygame.MOUSEBUTTONUP:
                click_loc = pygame.mouse.get_pos()

                input_point = np.array([list(click_loc)])
                input_label = np.array([1])

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                mask = masks[scores.argmax()]

                mask = np.stack([mask]*3,axis=2).astype(float)
                np.save('mask.npy', mask)

                g = 0.6
                bg = Image.fromarray((im * ((1-g) + g * mask.astype(float))).astype(np.uint8))

                # Show mask + image, then just mask
                #show_image(bg)
                #sleep(1)

                bg = (shade * im * (1 - mask) + 255 * mask).astype(np.uint8)
                show_image(Image.fromarray(bg))

                getting_mask = False
        else:
            if event.type == pygame.QUIT:
                running = False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_RETURN:
                    torch.save(flow, 'flow.pth')
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                arrow_start = pygame.mouse.get_pos()
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                # Clear screen
                screen.fill(white)

                arrow_end = pygame.mouse.get_pos()

                # Calculate dx and dy from arrow_start to arrow_end
                dx = arrow_end[0] - arrow_start[0]
                dy = arrow_end[1] - arrow_start[1]

                # Get the image based on dx and dy
                flow_image, flow = get_image(arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1])
                # Mask Image
                image = (shade * im * (1 - mask) + flow_image * mask).astype(np.uint8)
                image = Image.fromarray(image)
                # Convert PIL Image to Pygame surface
                image_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
                # Draw the new background
                screen.blit(image_surface, (0, 0))

                # Draw arrow while dragging
                if MODE == 'translate':
                    draw_arrow(screen, black, arrow_start, pygame.mouse.get_pos(), 3)
                elif MODE == 'rotate':
                    rotation_end = (
                            int(arrow_end[0] + dy / 2.),
                            int(arrow_end[1] - dx / 2.)
                                )
                    draw_arrow(screen, black, arrow_end, rotation_end, 3)
                    pygame.draw.circle(screen, black, arrow_start, 10, width=3)
                elif MODE == 'scale':
                    dr = math.sqrt(dx**2+dy**2)
                    start = (arrow_start[0] + int(dx / dr * 100), arrow_start[1] + int(dy / dr * 100))
                    draw_arrow(screen, black, start, pygame.mouse.get_pos(), 3)
                    pygame.draw.circle(screen, black, arrow_start, 100, width=3)
                elif MODE == 'scale_1d':
                    draw_arrow(screen, black, arrow_start, pygame.mouse.get_pos(), 3)

                    dr = math.sqrt(dx**2+dy**2)
                    ldx = -int(dy / dr * 10)
                    ldy = int(dx / dr * 10)
                    udx = int(dx / dr * 100)
                    udy = int(dy / dr * 100)
                    start = (arrow_start[0] + ldx, arrow_start[1] + ldy)
                    end = (arrow_start[0] - ldx, arrow_start[1] - ldy)
                    pygame.draw.line(screen, black, start, end, 3)

                    start = (arrow_start[0] + udx + ldx, arrow_start[1] + udy + ldy)
                    end = (arrow_start[0] + udx - ldx, arrow_start[1] + udy - ldy)
                    pygame.draw.line(screen, black, start, end, 3)

                pygame.display.flip()

# Quit Pygame
pygame.quit()


