import os
from cntk import load_model
from cntk.ops import softmax
import pandas as pd
import numpy as np
from PIL import Image


# define base model location and characteristics
_image_height = 512
_image_width = 682
_num_channels = 3


class ModelLoader:
    def __init__(self):
        self.model_list = {'21C': None, '3C': None}
        model = Model()
        self.model_list['DummyTest'] = model
        return

    def get_model(self, model_type):
        model = self.model_list.get(model_type)
        if model is None:
            model = CntkModel(model_type)
            if model.load():
                self.model_list[model_type] = model
            else:
                print("Failed to load CNTK model:")
                model = None
        return model


class Model:
    def __init__(self):
        self.name = 'Dummy Test Model'
        self.base_model = 'Unknown'
        self.model_path = os.path.join(os.getcwd(), 'models')
        self.model_file = None
        return

    def load(self):
        print(" Dummy test model load()")
        return

    def predict(self, image_path):
        print(" Dummy test predict()")
        return []


class CntkModel(Model):
    def __init__(self, model_type):
        super().__init__()
        self._base_model_name = "ResNet18_ImageNet_CNTK.model"
        self.name = "TNC_" + self._base_model_name
        self.model_path = os.path.join(self.model_path, model_type)
        self.model_file = os.path.join(self.model_path, self.name)
        self.trained_model = None
        self.lookup_file = os.path.join(self.model_path, "Label_ClassID_Lookup.csv")
        self.lookup_df = None
        return

    def load(self):
        self.lookup_df = pd.read_csv(self.lookup_file, index_col='ClassID', encoding='utf-8-sig')
        if self.lookup_df is None:
            print("Error: couldn't load look-up file - ", self.lookup_file)
            return None

        print(self.lookup_df.columns)

        self.trained_model = load_model(self.model_file)
        if self.trained_model is None:
            print("Error: couldn't load pre-trained model - ", self.model_file)
            return None
        print("CNTK model loaded: ", self.model_file)
        return self.trained_model

    def predict(self, image_path):
        prob = self._eval_single_image(image_path)
        # class_id = np.np.argmax(prob)
        self.lookup_df['probability'] = 0.0
        for i in range(prob.size):
            self.lookup_df.at[i, 'probability'] = prob[i]
        sorted_df = self.lookup_df.sort_values(by=['probability'], ascending=False)

        predictions = []
        for i in range(sorted_df.shape[0]):
            class_id = str(sorted_df.iloc[i].name)
            tag = sorted_df.iloc[i].Label
            p = str(sorted_df.iloc[i].probability)
            print("{!s}:{!s}".format(tag, p))
            predictions.append({"TagId": class_id,
                                "Tag": tag,
                                "Probability": p})
        return predictions

    # Evaluates a single image using the provided model
    def _eval_single_image(self, image_path, image_width=_image_width, image_height=_image_height):
        # load and format image (resize, RGB -> BGR, CHW -> HWC)
        img = Image.open(image_path)
        if image_path.endswith("png"):
            temp = Image.new("RGB", img.size, (255, 255, 255))
            temp.paste(img, img)
            img = temp
        re_sized = img.resize((image_width, image_height), Image.ANTIALIAS)
        bgr_image = np.asarray(re_sized, dtype=np.float32)[..., [2, 1, 0]]
        hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # Alternatively: if you want to use opencv-python
        # cv_img = cv2.imread(image_path)
        # resized = cv2.resize(cv_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        # bgr_image = np.asarray(resized, dtype=np.float32)
        # hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # compute model output
        arguments = {self.trained_model.arguments[0]: [hwc_format]}
        output = self.trained_model.eval(arguments)

        # return softmax probabilities
        sm = softmax(output[0])
        return sm.eval()
