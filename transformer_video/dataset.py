import math

import matplotlib.pyplot as plt
import numpy as np
import pymunk
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont


class Dataset:
    
    def __init__(
        self,
        image_height = 16,
        image_width = 16,
        shape_side_length = 2,
        fps = 60,
        
        speed_mean=5,
        speed_sd=2,
        gravity_mean=np.pi**2,
        gravity_sd=0,
        restitution_min=0.80,
        restitution_max=1.00,
        
        direction_min=0,
        direction_max=np.pi*2,
        position_x_mean=None,
        position_x_sd=1,
        position_y_mean=None,
        position_y_sd=1,
    ):
        
        self.image_height = image_height
        self.image_width = image_width
        self.shape_side_length = shape_side_length
        self.fps = fps
        
        self.speed_mean = speed_mean
        self.speed_sd = speed_sd
        self.gravity_mean = gravity_mean
        self.gravity_sd = gravity_sd
        self.restitution_min = restitution_min
        self.restitution_max = restitution_max
        
        self.direction_min = direction_min
        self.direction_max = direction_max
        self.position_x_mean = position_x_mean or image_width/2
        self.position_x_sd = position_x_sd
        self.position_y_mean = position_y_mean or image_height/2
        self.position_y_sd = position_y_sd
        
    
    def simulate_motion(self, initial_pos, velocity, gravity, restitution, time_step):
        """
        Simulates the motion of a square object within a bounded space using Pymunk physics engine.
        The function creates a 2D physics simulation, adds a square body with specified initial properties,
        and simulates its motion for a given time step.

        Args:
            initial_pos (tuple of float): The initial position (x, y) of the square.
            velocity (tuple of float): The initial velocity (vx, vy) of the square.
            gravity (float): The gravitational acceleration applied in the simulation.
                            Positive values pull the square downward.
            restitution (float): The elasticity coefficient of the square and boundaries.
                                Values are between 0 (perfectly inelastic) and 1 (perfectly elastic).
            time_step (float): The duration for which the simulation is advanced.

        Returns:
            tuple: A tuple containing the new position (x, y) and velocity (vx, vy) of the square after the simulation step.
        """
        # Create a new space and set gravity
        space = pymunk.Space()
        space.gravity = (0, -gravity)

        # Create a body and shape for the square
        body = pymunk.Body(
            1, pymunk.moment_for_box(1, (self.shape_side_length, self.shape_side_length))
        )
        body.position = pymunk.Vec2d(*initial_pos)  # Unpack the initial_pos tuple
        body.velocity = pymunk.Vec2d(*velocity)  # Unpack the velocity tuple
        shape = pymunk.Poly.create_box(body, (self.shape_side_length, self.shape_side_length))
        shape.elasticity = restitution
        space.add(body, shape)

        # Add static lines to form boundaries of the space
        static_lines = [
            pymunk.Segment(space.static_body, (0, 0), (0, self.image_height), 1),  # Left
            pymunk.Segment(
                space.static_body, (0, self.image_height), (self.image_width, self.image_height), 1
            ),  # Bottom
            pymunk.Segment(
                space.static_body, (self.image_width, self.image_height), (self.image_width, 0), 1
            ),  # Right
            pymunk.Segment(space.static_body, (self.image_width, 0), (0, 0), 1),  # Top
        ]
        for line in static_lines:
            line.elasticity = restitution  # Set restitution for the boundaries
            space.add(line)

        # Simulate for the given time step
        space.step(time_step)

        # Return the new position and velocity
        new_pos = body.position.x, body.position.y
        new_vel = body.velocity.x, body.velocity.y

        return new_pos, new_vel


    def draw_frame(self, position):
        """
        Draw a frame with the shape at the given position in black and white.
        """
        image = Image.new(
            "1", (self.image_width, self.image_height), "white"
        )  # '1' for 1-bit pixels, black and white
        draw = ImageDraw.Draw(image)

        # Draw the square shape in black|
        x, y = position

        draw.rectangle([x, y, x + self.shape_side_length, y + self.shape_side_length], fill="black")

        return image


    def generate_sequence(
        self,
        sequence_length,
        initial_speed,
        initial_direction,
        initial_position,
        gravity,
        coefficient_of_restitution,
        # frame_rate=30,
    ):
        """
        Generate a sequence of images of a square object moving in a bounded space.
        """
        images = []
        position = initial_position
        velocity = (
            initial_speed * np.cos(initial_direction),
            -initial_speed * np.sin(initial_direction),
        )

        for _ in range(sequence_length):
            for _ in range(self.fps):  # Advance the simulation frame_rate times before generating an image
                position, velocity = self.simulate_motion(
                    position,
                    velocity,
                    gravity,
                    coefficient_of_restitution,
                    1.0 / self.fps,
                    # 1.0 / 60,  # Assuming 60 FPS for the simulation
                )

            adjusted_position = (
                position[0],
                self.image_height - position[1] - self.shape_side_length,
            )
            image = self.draw_frame(adjusted_position)
            images.append(image)

        return images


    def generate_random_sequence(
        self,
        sequence_length,
        initial_speed, 
        gravity, 
        coefficient_of_restitution
    ):
        """
        Generate a sequence of images of a square object moving in a bounded space with random initial properties.
        """
        # Sample each parameter
        initial_direction = np.random.uniform(self.direction_min, self.direction_max)
        initial_position_x = np.random.normal(self.position_x_mean, self.position_x_sd)
        initial_position_y = np.random.normal(self.position_y_mean, self.position_y_sd)
        
        initial_position_x = min(max(initial_position_x, 0), self.image_width)
        initial_position_y = min(max(initial_position_y, 0), self.image_height)
        
        if initial_position_x in (0, self.image_width):
            print('X was out of clipped for being out of bounds')
        if initial_position_y in (0, self.image_height):
            print('Y was out of clipped for being out of bounds')

        # Generate the sequence
        sequence = self.generate_sequence(
            sequence_length,
            initial_speed,
            initial_direction,
            (initial_position_x, initial_position_y),
            gravity,
            coefficient_of_restitution,
        )

        return sequence
    
    def query(
        self,
        samples=3,
        sequence_length=10,
    ):
        initial_speed = np.random.normal(self.speed_mean, self.speed_sd)
        gravity = np.random.normal(self.gravity_mean, self.gravity_sd)
        coefficient_of_restitution = np.random.uniform(self.restitution_min, self.restitution_max)
        
        out = []
        for _ in range(samples):
            seq = self.generate_random_sequence(sequence_length, initial_speed, gravity, coefficient_of_restitution)
            out.append(seq)
        return out
        

    def display_sequence(self, sequence):
        # Display the images side by side with boundaries between frames
        fig, axes = plt.subplots(
            1, len(sequence), figsize=(20, 2)
        )  # Adjust figsize as needed

        # Adding a small space between images for clear separation
        plt.subplots_adjust(wspace=0.1)  # Adjust space as needed

        for ax, img in zip(axes, sequence):
            ax.imshow(img)
            ax.axis("on")  # Turn on axis to create a boundary
            ax.set_xticks([])
            ax.set_yticks([])  # Remove tick marks

        plt.show()


    def create_question_mark_image(self, image_width, image_height, font_size):
        # Create a new image with white background
        image = Image.new("1", (image_width, image_height), "white")

        # Prepare to draw on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        # Calculate the position for the question mark to be centered
        text = "?"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (image_width - text_width) / 2
        y = (image_height - text_height) / 2

        # Draw the question mark on the image
        draw.text((x, y), text, fill="black", font=font)

        return image