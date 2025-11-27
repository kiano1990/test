"""
# Readme


# License Plate Recognition

This example demonstrates how to run 3 stage inference on DepthAI. Its focus is on an automatic license plate recognition (ALPR) task.
First, a vehicle is detected on the image, the cropped image is then fed into a license plate detection model. The cropped license plate is sent to a text recognition (OCR) network, which tries to decode the license plates texts.

It uses 3 models from our ZOO:

- [YOLOv6 nano](https://models.luxonis.com/luxonis/yolov6-nano/face58c4-45ab-42a0-bafc-19f9fee8a034) for vehicle detection.
- [License Plate Detection](https://models.luxonis.com/luxonis/license-plate-detection/7ded2dab-25b4-4998-9462-cba2fcc6c5ef) for detecting the license plates.
- [PaddlePaddle Rext Recognition](https://models.luxonis.com/luxonis/paddle-text-recognition/9ae12b58-3551-49b1-af22-721ba4bcf269) for recognizing text on license plates.

**NOTE**: Due to the high computational cost, this example only works on OAK4 devices.

Take a look at [How to Train and Deploy a License Plate Detector to the Luxonis OAK](https://blog.roboflow.com/oak-deploy-license-plate/) tutorial for training a custom detector using the Roboflow platform.

## Demo

![Detection Output](media/lpr.gif)

<sup>[Source](https://www.pexels.com/video/speeding-multicolored-cars-trucks-and-suv-motor-vehicles-exit-a-dark-new-york-city-underground-tunnel-which-is-wrapped-in-the-lush-green-embrace-of-trees-and-bushes-17108719/)</sup>

## Usage

Running this example requires a **Luxonis device** connected to your computer. Refer to the [documentation](https://docs.luxonis.com/software-v3/) to setup your device if you haven't done it already.

You can run the example fully on device ([`STANDALONE` mode](#standalone-mode-rvc4-only)) or using your computer as host ([`PERIPHERAL` mode](#peripheral-mode)).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

## Peripheral Mode

### Installation

You need to first prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

### Examples

```bash
python3 main.py
```

This will run the example with the default device and camera input.

## Standalone Mode (RVC4 only)

Running the example in the standalone mode, app runs entirely on the device.
To run the example in this mode, first install the `oakctl` tool using the installation instructions [here](https://docs.luxonis.com/software-v3/oak-apps/oakctl).

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the example with default argument values. If you want to change these values you need to edit the `oakapp.toml` file (refer [here](https://docs.luxonis.com/software-v3/oak-apps/configuration/) for more information about this configuration file).







"""
########################################################################

# utils -> config_sender_scrupt.py


try:
    while True:
        frame = node.inputs["frame_input"].get()
        node.warn("got frame")
        detections_message = node.inputs["detections_input"].get()
        node.warn("got detections")
        vehicle_detections = []

        for d in detections_message.detections:
            if d.label not in [2, 5, 7]:  # not a car, bus or truck
                continue
            if d.confidence < 0.7:
                continue
            d.xmin = max(0, min(1, d.xmin * 0.9))
            d.ymin = max(0, min(1, d.ymin * 0.9))
            d.xmax = max(0, min(1, d.xmax * 1.1))
            d.ymax = max(0, min(1, d.ymax * 1.1))

            vehicle_detections.append(d)

            x_center = (d.xmin + d.xmax) / 2
            y_center = (d.ymin + d.ymax) / 2
            det_w = d.xmax - d.xmin
            det_h = d.ymax - d.ymin

            det_center = Point2f(x_center, y_center, normalized=True)
            det_size = Size2f(det_w, det_h, normalized=True)

            det_rect = RotatedRect(det_center, det_size, 0)
            det_rect = det_rect.denormalize(frame.getWidth(), frame.getHeight())

            cfg = ImageManipConfig()
            cfg.addCropRotatedRect(det_rect, normalizedCoords=False)
            cfg.setOutputSize(640, 640)
            cfg.setReusePreviousImage(False)
            cfg.setTimestamp(detections_message.getTimestamp())
            node.outputs["output_config"].send(cfg)
            node.outputs["output_frame"].send(frame)

        vehicle_detections_msg = ImgDetections()
        vehicle_detections_msg.detections = vehicle_detections
        vehicle_detections_msg.setTimestamp(detections_message.getTimestamp())
        vehicle_detections_msg.setTransformation(detections_message.getTransformation())

        node.warn("sending vehicle crop config")
        node.outputs["output_vehicle_detections"].send(vehicle_detections_msg)

except Exception as e:
    node.warn(str(e))
############################################################################################################

# utils -> license_plate_sender_script.py


def denormalize_detection(detection: ImgDetection, width: int, height: int):
    x_min, y_min, x_max, y_max = (
        detection.xmin,
        detection.ymin,
        detection.xmax,
        detection.ymax,
    )

    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)

    return x_min, y_min, x_max, y_max


try:
    while True:
        frame = node.inputs["frame_input"].get()
        frame_w = frame.getWidth()
        frame_h = frame.getHeight()

        detections_message = node.inputs["detections_input"].get()
        detections = detections_message.detections

        valid_detections = []
        valid_crops = []
        for d in detections:
            x_min, y_min, x_max, y_max = denormalize_detection(d, frame_w, frame_h)
            w = x_max - x_min
            h = y_max - y_min

            license_plate_detections = (
                node.inputs["license_plate_detections"].get().detections
            )

            if len(license_plate_detections) == 0:
                continue

            license_plate_detection = sorted(
                license_plate_detections, key=lambda x: x.confidence, reverse=True
            )[0]
            if license_plate_detection.confidence < 0.5:
                continue

            license_plate_detection.xmin = license_plate_detection.xmin * 1.02
            license_plate_detection.ymin = license_plate_detection.ymin * 1.03
            license_plate_detection.xmax = license_plate_detection.xmax * 0.98
            license_plate_detection.ymax = license_plate_detection.ymax * 0.97

            lp_x_min, lp_y_min, lp_x_max, lp_y_max = denormalize_detection(
                license_plate_detection, w, h
            )
            lp_w = lp_x_max - lp_x_min
            lp_h = lp_y_max - lp_y_min

            crop_x_min = max(0, min(lp_x_min + x_min, frame_w))
            crop_y_min = max(0, min(lp_y_min + y_min, frame_h))
            crop_w = max(0, min(lp_w, frame_w - crop_x_min))
            crop_h = max(0, min(lp_h, frame_h - crop_y_min))

            if crop_w <= 40 or crop_h <= 10:
                continue

            cfg = ImageManipConfig()
            cfg.addCrop(crop_x_min, crop_y_min, crop_w, crop_h)
            cfg.setReusePreviousImage(False)
            cfg.setOutputSize(320, 48)
            cfg.setTimestamp(detections_message.getTimestamp())

            valid_detections.append(d)
            valid_crops.append(license_plate_detection)
            node.outputs["lp_crop_config"].send(cfg)
            node.outputs["lp_crop_frame"].send(frame)

        valid_detections_msg = ImgDetections()
        valid_detections_msg.detections = valid_detections
        valid_detections_msg.setTimestamp(detections_message.getTimestamp())

        valid_crops_msg = ImgDetections()
        valid_crops_msg.detections = valid_crops
        valid_crops_msg.setTimestamp(detections_message.getTimestamp())
        node.warn("sending vehicle crop config")
        node.outputs["output_valid_crops"].send(valid_crops_msg)
        node.outputs["output_valid_detections"].send(valid_detections_msg)

except Exception as e:
    node.warn(str(e))
###################################################################################################


# utils -> visualizer_node.py

import cv2
import depthai as dai
import numpy as np


class VisualizeLicensePlates(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()

        self.vehicle_detections = self.createInput()
        self.input_frame = self.createInput()
        self.ocr_results = self.createInput()
        self.lp_crop_detections = self.createInput()
        self.lp_crop_images = self.createInput()

        self.out = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            frame_message = self.input_frame.get()
            frame = frame_message.getCvFrame()
            frame_h, frame_w = frame.shape[:2]

            detections = self.vehicle_detections.get().detections
            crop_detections = self.lp_crop_detections.get().detections

            for detection, lp_detection in zip(detections, crop_detections):
                x_min = int(detection.xmin * frame_w)
                y_min = int(detection.ymin * frame_h)
                x_max = int(detection.xmax * frame_w)
                y_max = int(detection.ymax * frame_h)

                vehicle_w = x_max - x_min
                vehicle_h = y_max - y_min

                ocr_message = self.ocr_results.get()
                text = "".join(ocr_message.classes)
                license_plate = self.lp_crop_images.get().getCvFrame()

                if len(text) < 5:
                    continue

                lp_x_min = int(lp_detection.xmin * vehicle_w) + x_min
                lp_y_min = int(lp_detection.ymin * vehicle_h) + y_min
                lp_x_max = int(lp_detection.xmax * vehicle_w) + x_min
                lp_y_max = int(lp_detection.ymax * vehicle_h) + y_min

                lp_x_min = np.clip(lp_x_min, 0, frame_w)
                lp_y_min = np.clip(lp_y_min, 0, frame_h)
                lp_x_max = np.clip(lp_x_max, 0, frame_w)
                lp_y_max = np.clip(lp_y_max, 0, frame_h)

                license_plate = cv2.resize(license_plate, (80, 12))

                white_frame = np.ones((12, 80, 3)) * 255
                cv2.putText(
                    white_frame,
                    text,
                    (2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
                crop_region = frame[lp_y_max : lp_y_max + 24, lp_x_min : lp_x_min + 80]
                lp_text = np.concatenate((license_plate, white_frame), axis=0)
                lp_text = cv2.resize(
                    lp_text, (crop_region.shape[1], crop_region.shape[0])
                )

                frame[
                    lp_y_max : lp_y_max + crop_region.shape[0],
                    lp_x_min : lp_x_min + crop_region.shape[1],
                ] = lp_text

            ts = frame_message.getTimestamp()
            frame_type = frame_message.getType()
            img = dai.ImgFrame()
            img.setCvFrame(frame, frame_type)
            img.setTimestamp(ts)
            img.setTimestampDevice(frame_message.getTimestampDevice())
            self.out.send(img)

############################################################################################################
#util -> arguments

import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = "OCR example script to show text and show it on a white background. \
        All you need is an OAK device and access to HubAI. Optionally, you can also run the model on a media file."

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "-media",
        "--media_path",
        help="Path to the media file you aim to run the model on. If not set, the model will run on the camera input.",
        required=False,
        default=None,
        type=str,
    )
    args = parser.parse_args()

    return parser, args

################################################################################################################


# main.py

from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

from utils.arguments import initialize_argparser
from utils.visualizer_node import VisualizeLicensePlates

REQ_WIDTH, REQ_HEIGHT = (
    1920 * 2,
    1080 * 2,
)

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

if platform != "RVC4":
    raise ValueError("This example is only supported for RVC4 platform.")

frame_type = dai.ImgFrame.Type.BGR888i

if args.fps_limit is None:
    args.fps_limit = 25
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # vehicle detection model
    vehicle_det_model_description = dai.NNModelDescription.fromYamlFile(
        f"yolov6_nano_r2_coco.{platform}.yaml"
    )
    vehicle_det_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(vehicle_det_model_description)
    )
    vehicle_det_model_w, vehicle_det_model_h = (
        vehicle_det_model_nn_archive.getInputSize()
    )

    # licence plate detection model
    lp_det_model_description = dai.NNModelDescription.fromYamlFile(
        f"license_plate_detection.{platform}.yaml"
    )
    lp_det_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(lp_det_model_description)
    )
    lp_det_model_w, lp_det_model_h = lp_det_model_nn_archive.getInputSize()

    # ocr model
    ocr_model_description = dai.NNModelDescription.fromYamlFile(
        f"paddle_text_recognition.{platform}.yaml"
    )
    ocr_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(ocr_model_description))
    ocr_model_w, ocr_model_h = ocr_model_nn_archive.getInputSize()

    # media/camera input
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        replay_resize = pipeline.create(dai.node.ImageManip)
        replay_resize.initialConfig.setOutputSize(REQ_WIDTH, REQ_HEIGHT)
        replay_resize.initialConfig.setReusePreviousImage(False)
        replay_resize.setMaxOutputFrameSize(REQ_WIDTH * REQ_HEIGHT * 3)
        replay.out.link(replay_resize.inputImage)
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            (REQ_WIDTH, REQ_HEIGHT), frame_type, fps=args.fps_limit
        )
    input_node_out = replay_resize.out if args.media_path else cam_out

    # resize input to vehicle det model input size
    vehicle_det_resize_node = pipeline.create(dai.node.ImageManip)
    vehicle_det_resize_node.initialConfig.setOutputSize(
        vehicle_det_model_w, vehicle_det_model_h
    )
    vehicle_det_resize_node.initialConfig.setReusePreviousImage(False)
    input_node_out.link(vehicle_det_resize_node.inputImage)

    vehicle_det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_det_resize_node.out, vehicle_det_model_nn_archive
    )

    # process vehicle detections
    config_sender_node = pipeline.create(dai.node.Script)
    config_sender_node.setScriptPath(
        Path(__file__).parent / "utils/config_sender_script.py"
    )
    config_sender_node.setLogLevel(dai.LogLevel.CRITICAL)

    input_node_out.link(config_sender_node.inputs["frame_input"])
    vehicle_det_nn.out.link(config_sender_node.inputs["detections_input"])

    vehicle_crop_node = pipeline.create(dai.node.ImageManip)
    vehicle_crop_node.initialConfig.setReusePreviousImage(False)
    vehicle_crop_node.inputConfig.setReusePreviousMessage(False)
    vehicle_crop_node.inputImage.setReusePreviousMessage(False)
    vehicle_crop_node.setMaxOutputFrameSize(lp_det_model_w * lp_det_model_h * 3)

    config_sender_node.outputs["output_config"].link(vehicle_crop_node.inputConfig)
    config_sender_node.outputs["output_frame"].link(vehicle_crop_node.inputImage)

    # per vehicle license plate detection
    lp_config_sender = pipeline.create(dai.node.Script)
    lp_config_sender.setScriptPath(
        Path(__file__).parent / "utils/license_plate_sender_script.py"
    )
    lp_config_sender.setLogLevel(dai.LogLevel.CRITICAL)

    input_node_out.link(lp_config_sender.inputs["frame_input"])

    lp_det_nn = pipeline.create(ParsingNeuralNetwork).build(
        vehicle_crop_node.out, lp_det_model_nn_archive
    )
    config_sender_node.outputs["output_vehicle_detections"].link(
        lp_config_sender.inputs["detections_input"]
    )
    lp_det_nn.out.link(lp_config_sender.inputs["license_plate_detections"])

    # resize detected licence plates to ocr model input size
    lp_crop_node = pipeline.create(dai.node.ImageManip)
    vehicle_crop_node.initialConfig.setReusePreviousImage(False)
    lp_crop_node.inputConfig.setReusePreviousMessage(False)
    lp_crop_node.inputImage.setReusePreviousMessage(False)
    lp_crop_node.setMaxOutputFrameSize(ocr_model_w * ocr_model_h * 3)

    lp_config_sender.outputs["lp_crop_config"].link(lp_crop_node.inputConfig)
    lp_config_sender.outputs["lp_crop_frame"].link(lp_crop_node.inputImage)

    ocr_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        lp_crop_node.out, ocr_model_nn_archive
    )
    ocr_nn.getParser(0).setIgnoredIndexes(
        [
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            44,
            45,
            46,
            47,
            48,
            49,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            93,
            94,
            95,
            96,
        ]
    )

    # annotation
    visualizer_node = pipeline.create(VisualizeLicensePlates)
    lp_config_sender.outputs["output_valid_detections"].link(
        visualizer_node.vehicle_detections
    )
    vehicle_det_nn.passthrough.link(visualizer_node.input_frame)
    ocr_nn.out.link(visualizer_node.ocr_results)
    lp_config_sender.outputs["output_valid_crops"].link(
        visualizer_node.lp_crop_detections
    )
    lp_crop_node.out.link(visualizer_node.lp_crop_images)

    # visualization
    visualizer.addTopic("License Plates", visualizer_node.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
