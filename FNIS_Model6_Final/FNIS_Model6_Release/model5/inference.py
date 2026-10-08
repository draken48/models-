
import json
import numpy as np
from PIL import Image

import onnxruntime as ort


class FNISModel5:

    def __init__(
        self,
        model_path="model5_efficientnetv2s_best.onnx",
        config_path="config.json",
        classes_path="classes.txt"
    ):

        with open(
            config_path,
            "r"
        ) as f:
            self.config = json.load(f)

        self.classes = {}

        with open(
            classes_path,
            "r"
        ) as f:

            for line in f:

                idx, name = line.rstrip("\n").split(
                    "\t",
                    1
                )

                self.classes[int(idx)] = name

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

        self.mean = np.asarray(
            self.config["normalization"]["mean"],
            dtype=np.float32
        ).reshape(1, 3, 1, 1)

        self.std = np.asarray(
            self.config["normalization"]["std"],
            dtype=np.float32
        ).reshape(1, 3, 1, 1)

    def preprocess(self, image):

        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.open(
                image
            )

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            (400, 400),
            Image.Resampling.BILINEAR
        )

        # Center crop 384 x 384.
        left = (400 - 384) // 2
        top = (400 - 384) // 2

        image = image.crop(
            (
                left,
                top,
                left + 384,
                top + 384
            )
        )

        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        array = np.transpose(
            array,
            (2, 0, 1)
        )

        array = array[
            np.newaxis,
            ...
        ]

        array = (
            array - self.mean
        ) / self.std

        return array.astype(
            np.float32
        )

    def _predict_single(
        self,
        array
    ):

        logits = self.session.run(
            [self.output_name],
            {
                self.input_name: array
            }
        )[0]

        logits = logits.astype(
            np.float32
        )

        logits -= logits.max(
            axis=1,
            keepdims=True
        )

        probabilities = np.exp(
            logits
        )

        probabilities /= (
            probabilities.sum(
                axis=1,
                keepdims=True
            )
        )

        return probabilities[0]

    def predict(
        self,
        image,
        use_tta=True
    ):

        array = self.preprocess(
            image
        )

        if not use_tta:

            probabilities = (
                self._predict_single(
                    array
                )
            )

        else:

            flipped = np.flip(
                array,
                axis=3
            ).copy()

            p1 = self._predict_single(
                array
            )

            p2 = self._predict_single(
                flipped
            )

            probabilities = (
                p1 + p2
            ) / 2.0

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        return {
            "class_index": predicted_index,

            "class_name":
                self.classes[predicted_index],

            "confidence":
                float(
                    probabilities[
                        predicted_index
                    ]
                ),

            "probabilities": {
                self.classes[i]:
                    float(probabilities[i])
                for i in range(
                    len(probabilities)
                )
            }
        }
