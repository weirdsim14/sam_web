import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Set the stream configuration
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# After starting the pipeline, add these lines to set the exposure and white balance
device = pipeline.get_active_profile().get_device()
color_sensor = device.query_sensors()[1]  # Get the color sensor
color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto exposure
color_sensor.set_option(rs.option.exposure, 50)  # Set exposure value
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)  # Disable auto white balance
color_sensor.set_option(rs.option.white_balance, 4500)  # Set white balance value

# Capture one color frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# Stop streaming
pipeline.stop()

# Convert the frame data to a NumPy array
color_image = np.asanyarray(color_frame.get_data())

# Save the color image as a JPG file
cv2.imwrite('color_image.jpg', color_image)
