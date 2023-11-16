import math

import matplotlib.pyplot as plt
import numpy as np
import pymunk
import torch
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont


class Dataset:
    def __init__(
        self,
        image_height=32,
        image_width=32,
        shape_side_length=2,
        fps=30,
        speed_mean=3,
        speed_sd=1,
        gravity_mean=0,
        # gravity_mean=np.pi**2,
        gravity_sd=0,
        restitution_min=1.00,
        restitution_max=1.00,
        direction_min=0,
        direction_max=np.pi * 2,
        position_x_mean=None,
        position_x_sd=2,
        position_y_mean=None,
        position_y_sd=2,
        mass=1.0,
        invert_colors=False,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.shape_side_length = shape_side_length
        self.fps = fps  # FPS is essentially our sampling rate, and thus effectively a factor effecting the speed of the object

        self.speed_mean = speed_mean
        self.speed_sd = speed_sd
        self.gravity_mean = gravity_mean
        self.gravity_sd = gravity_sd
        self.restitution_min = restitution_min
        self.restitution_max = restitution_max

        self.direction_min = direction_min
        self.direction_max = direction_max
        self.position_x_mean = position_x_mean or image_width / 2
        self.position_x_sd = position_x_sd
        self.position_y_mean = position_y_mean or image_height / 2
        self.position_y_sd = position_y_sd
        self.mass = mass  # I'm fairly sure this doesn't matter, since we don't incorporate any rotation so momentum doesn't matter
        self.invert_colors = invert_colors  # perhaps this impacts training?

    def simulate_motion(self, initial_pos, velocity, gravity, restitution):
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

        Returns:
            tuple: A tuple containing the new position (x, y) and velocity (vx, vy) of the square after the simulation step.
        """
        # Create a new space and set gravity
        space = pymunk.Space()
        space.gravity = (0, -gravity)

        # Create a body and shape for the square
        body = pymunk.Body(
            self.mass,
            pymunk.moment_for_box(
                self.mass, (self.shape_side_length, self.shape_side_length)
            ),
        )
        body.position = pymunk.Vec2d(*initial_pos)  # Unpack the initial_pos tuple
        body.velocity = pymunk.Vec2d(*velocity)  # Unpack the velocity tuple
        shape = pymunk.Poly.create_box(
            body, (self.shape_side_length, self.shape_side_length)
        )
        shape.elasticity = restitution
        space.add(body, shape)

        # Add static lines to form boundaries of the space
        left = pymunk.Segment(space.static_body, (0, 0), (0, self.image_height), 1)
        bottom = pymunk.Segment(
            space.static_body,
            (0, self.image_height),
            (self.image_width, self.image_height),
            1,
        )
        top = pymunk.Segment(space.static_body, (self.image_width, 0), (0, 0), 1)
        right = pymunk.Segment(
            space.static_body,
            (self.image_width, self.image_height),
            (self.image_width, 0),
            1,
        )

        for line in [left, bottom, right, top]:
            line.elasticity = restitution  # Set restitution for the boundaries
            space.add(line)

        # Simulate for the given time step
        time_step = 1.0 / self.fps
        space.step(time_step)

        # Return the new position and velocity
        new_pos = body.position.x, body.position.y
        new_vel = body.velocity.x, body.velocity.y

        return new_pos, new_vel

    def draw_frame(self, position):
        """
        Draw a frame with the shape at the given position in black and white.
        """
        # '1' for 1-bit pixels, black and white
        img_color, ball_color = (
            ("black", "white") if self.invert_colors else ("white", "black")
        )

        image = Image.new("1", (self.image_width, self.image_height), img_color)
        draw = ImageDraw.Draw(image)

        # Draw the square shape in black|
        x, y = position

        if self.shape_side_length == 3:
            draw.rectangle(
                [
                    x - 1,
                    y - 1,
                    x + self.shape_side_length + 1,
                    y + self.shape_side_length + 1,
                ],
                fill=ball_color,
            )
        else:
            draw.rectangle(
                [x, y, x + self.shape_side_length, y + self.shape_side_length],
                fill=ball_color,
            )

        image = np.asarray(image)
        image = np.expand_dims(image, axis=2)
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
            for _ in range(self.fps):
                position, velocity = self.simulate_motion(
                    position,
                    velocity,
                    gravity,
                    coefficient_of_restitution,
                )

            adjusted_position = (
                position[0],
                self.image_height - position[1] - self.shape_side_length,
            )
            image = self.draw_frame(adjusted_position)
            images.append(image)

        images = np.asarray(images)
        return images

    def generate_random_sequence(
        self, sequence_length, initial_speed, gravity, coefficient_of_restitution
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
            print("X was out of clipped for being out of bounds")
        if initial_position_y in (0, self.image_height):
            print("Y was out of clipped for being out of bounds")

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
        as_tensor=True,
        seed=None,
    ):
        seed = seed or np.random.randint(0, 1000000)
        np.random.seed(seed)

        initial_speed = np.random.normal(self.speed_mean, self.speed_sd)
        gravity = np.random.normal(self.gravity_mean, self.gravity_sd)
        coefficient_of_restitution = np.random.uniform(
            self.restitution_min, self.restitution_max
        )

        seqs = []
        for _ in range(samples):
            seq = self.generate_random_sequence(
                sequence_length, initial_speed, gravity, coefficient_of_restitution
            )
            seq = np.asarray(seq)
            if as_tensor:
                seq = torch.from_numpy(seq)
                # seq = ToTensor()(seq)

            seqs.append(seq)

        out = dict(
            sequences=seqs,
            speed=initial_speed,
            gravity=gravity,
            restitution=coefficient_of_restitution,
        )
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
