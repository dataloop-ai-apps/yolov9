from ultralytics import YOLO
from PIL import Image
import dtlpy as dl
import logging
import torch
import PIL
import os
import yaml
import shutil
import numpy as np

logger = logging.getLogger('YOLOv9Adapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000


@dl.Package.decorators.module(description='Model Adapter for Yolov9 object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):

    def save(self, local_path, **kwargs):
        self.configuration.update({'weights_filename': 'weights/best.pt'})

    def convert_from_dtlpy(self, data_path, **kwargs):
        ##############
        # Validation #
        ##############

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError(
                'Couldnt find train set. Yolov9 requires train and validation set for training. Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError(
                'Couldnt find validation set. Yolov9 requires train and validation set for training. Add a validation set DQL filter in the dl.Model metadata')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            if self.model_entity.output_type == 'box':
                filters.add_join(field='type', values='box')
            elif self.model_entity.output_type in ['segment', 'binary']:
                filters.add_join(field='type', values=['segment', 'binary'], operator=dl.FILTERS_OPERATIONS_IN)
            filters.page_size = 0
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f"Couldn't find box or segment annotations in subset {subset}. "
                    f"Cannot train without annotation in the data subsets")

        #########
        # Paths #
        #########

        train_path = os.path.join(data_path, 'train', 'json')
        validation_path = os.path.join(data_path, 'validation', 'json')
        label_to_id_map = self.model_entity.label_to_id_map

        #################
        # Convert Train #
        #################
        converter = dl.utilities.converter.Converter()
        converter.labels = label_to_id_map
        converter.convert_directory(local_path=train_path,
                                    dataset=self.model_entity.dataset,
                                    to_format='yolo',
                                    from_format='dataloop')
        ######################
        # Convert Validation #
        ######################
        converter = dl.utilities.converter.Converter()
        converter.labels = label_to_id_map
        converter.convert_directory(local_path=validation_path,
                                    dataset=self.model_entity.dataset,
                                    to_format='yolo',
                                    from_format='dataloop')

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', 'yolov9c.pt')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model_filepath = os.path.normpath(os.path.join(local_path, model_filename))

        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath)  # pass any model type
        elif os.path.isfile('/tmp/app/weights/' + model_filename):
            model = YOLO('/tmp/app/weights/' + model_filename)
        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/' + model_filename
            model = YOLO(url)  # pass any model type
        model.to(device=device)
        logger.info(f"Model loaded successfully, Device: {model.device}")
        self.confidence_threshold = self.configuration.get('conf_thres', 0.25)
        self.model = model
        self.update_tracker_configs()

    def update_tracker_configs(self):
        botsort_configs = self.configuration.get('botsort_configs', dict())
        # Load the YAML file
        with open('botsort.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # Edit existing keys/values
        data['track_high_thresh'] = botsort_configs.get('track_high_thresh', 0.25)
        data['track_low_thresh'] = botsort_configs.get('track_low_thresh', 0.1)
        data['new_track_thresh'] = botsort_configs.get('new_track_thresh', 0.5)
        data['track_buffer'] = botsort_configs.get('track_buffer', 30)
        data['match_thresh'] = botsort_configs.get('match_thresh', 0.8)

        # Write the updated data back to a YAML file
        with open('custom_botsort.yaml', 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)

    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        if 'image' in item.mimetype:
            data = Image.open(filename)
            # Check if the image has EXIF data
            if hasattr(data, '_getexif'):
                exif_data = data._getexif()
                # Get the EXIF orientation tag (if available)
                if exif_data is not None:
                    orientation = exif_data.get(0x0112)
                    if orientation is not None:
                        # Rotate the image based on the orientation tag
                        if orientation == 3:
                            data = data.rotate(180, expand=True)
                        elif orientation == 6:
                            data = data.rotate(270, expand=True)
                        elif orientation == 8:
                            data = data.rotate(90, expand=True)
            data = data.convert('RGB')
        else:
            data = filename
        return data, item

    def predict(self, batch, **kwargs):
        include_untracked = self.configuration.get('botsort_configs', dict()).get('include_untracked', False)
        batch_annotations = list()
        for stream, item in batch:
            track_ids = list(range(1000, 10001))
            if 'image' in item.mimetype:
                image_annotations = dl.AnnotationCollection()
                results = self.model.predict(source=stream, save=False, save_txt=False)  # save predictions as labels
                for i_img, res in enumerate(results):  # per image
                    if self.model_entity.output_type == 'segment' and res.masks:
                        for box, mask in zip(reversed(res.boxes), reversed(res.masks)):
                            cls, conf = box.cls.squeeze(), box.conf.squeeze()
                            c = int(cls)
                            label = res.names[c]
                            if label not in list(self.configuration.get("label_to_id_map", {}).keys()):
                                logger.error(f"Predict label {label} is not among the models' labels.")
                            image_annotations.add(annotation_definition=dl.Polygon(geo=mask.xy[0], label=label),
                                                  model_info={'name': self.model_entity.name,
                                                              'model_id': self.model_entity.id,
                                                              'confidence': float(conf)})
                    elif self.model_entity.output_type == 'binary' and res.masks:
                        for box, mask in zip(reversed(res.boxes), reversed(res.masks)):
                            cls, conf = box.cls.squeeze(), box.conf.squeeze()
                            c = int(cls)
                            label = res.names[c]
                            if label not in list(self.configuration.get("label_to_id_map", {}).keys()):
                                logger.error(f"Predict label {label} is not among the models' labels.")
                            image_annotations.add(annotation_definition=dl.Segmentation(geo=mask.xy[0], label=label),
                                                  model_info={'name': self.model_entity.name,
                                                              'model_id': self.model_entity.id,
                                                              'confidence': float(conf)})
                    elif self.model_entity.output_type == 'box':
                        for d in reversed(res.boxes):
                            cls = int(d.cls.squeeze())
                            conf = float(d.conf.squeeze())
                            if conf < self.confidence_threshold:
                                continue
                            label = res.names[cls]
                            xyxy = d.xyxy.squeeze()
                            image_annotations.add(annotation_definition=dl.Box(left=float(xyxy[0]),
                                                                               top=float(xyxy[1]),
                                                                               right=float(xyxy[2]),
                                                                               bottom=float(xyxy[3]),
                                                                               label=label
                                                                               ),
                                                  model_info={'name': self.model_entity.name,
                                                              'model_id': self.model_entity.id,
                                                              'confidence': conf})
                batch_annotations.append(image_annotations)
            if 'video' in item.mimetype:
                image_annotations = item.annotations.builder()
                results = self.model.track(source=stream,
                                           tracker='custom_botsort.yaml',
                                           stream=True,
                                           verbose=True,
                                           save=False,
                                           save_txt=False)
                for idx, frame in enumerate(results):
                    for box in frame.boxes:
                        if box.is_track is False:
                            if include_untracked is False:
                                continue
                            else:
                                # Guarantee unique object_id
                                object_id = track_ids.pop()
                                # object_id = random.randint(1000, 10000)
                        else:
                            object_id = int(box.id.squeeze())
                        cls = int(box.cls.squeeze())
                        conf = float(box.conf.squeeze())
                        label = self.model.names[cls]
                        xyxy = box.xyxy.squeeze()
                        image_annotations.add(annotation_definition=dl.Box(left=float(xyxy[0]),
                                                                           top=float(xyxy[1]),
                                                                           right=float(xyxy[2]),
                                                                           bottom=float(xyxy[3]),
                                                                           label=label
                                                                           ),
                                              model_info={'name': self.model_entity.name,
                                                          'model_id': self.model_entity.id,
                                                          'confidence': conf},
                                              object_id=object_id,
                                              frame_num=idx
                                              )
                batch_annotations.append(image_annotations)
        return batch_annotations

    @staticmethod
    def copy_files(src_path, dst_path):
        subfolders = [x[0] for x in os.walk(src_path)]
        os.makedirs(dst_path, exist_ok=True)

        for subfolder in subfolders:
            for filename in os.listdir(subfolder):
                file_path = os.path.join(subfolder, filename)
                if os.path.isfile(file_path):
                    # Get the relative path from the source directory
                    relative_path = os.path.relpath(subfolder, src_path)
                    # Create a new file name with the relative path included
                    new_filename = f"{relative_path.replace(os.sep, '_')}_{filename}"
                    new_file_path = os.path.join(dst_path, new_filename)
                    shutil.copy(file_path, new_file_path)

    def train(self, data_path, output_path, **kwargs):
        self.model.model.args.update(self.configuration.get('modelArgs', dict()))
        epochs = self.configuration.get('epochs', 50)
        start_epoch = self.configuration.get('start_epoch', 0)
        batch_size = self.configuration.get('batch_size', 2)
        imgsz = self.configuration.get('imgsz', 640)
        device = self.configuration.get('device', None)
        augment = self.configuration.get('augment', True)
        yaml_config = self.configuration.get('yaml_config', dict())
        resume = start_epoch > 0
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        project_name = os.path.dirname(output_path)
        name = os.path.basename(output_path)

        # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories
        train_name = 'train'
        val_name = 'validation'
        src_images_path_train = os.path.join(data_path, 'train', 'items')
        dst_images_path_train = os.path.join(data_path, train_name, 'images')
        src_images_path_val = os.path.join(data_path, 'validation', 'items')
        dst_images_path_val = os.path.join(data_path, val_name, 'images')
        src_labels_path_train = os.path.join(data_path, 'train', 'yolo')
        dst_labels_path_train = os.path.join(data_path, train_name, 'labels')
        src_labels_path_val = os.path.join(data_path, 'validation', 'yolo')
        dst_labels_path_val = os.path.join(data_path, val_name, 'labels')

        # copy images and labels to train and validation directories
        self.copy_files(src_images_path_train, dst_images_path_train)  # add dir to name
        self.copy_files(src_images_path_val, dst_images_path_val)
        self.copy_files(src_labels_path_train, dst_labels_path_train)
        self.copy_files(src_labels_path_val, dst_labels_path_val)

        # check if validation exists
        if not os.path.isdir(dst_images_path_val):
            raise ValueError(
                'Couldnt find validation set. Yolov9 requires train and validation set for training. Add a validation set DQL filter in the dl.Model metadata')
        if len(self.model_entity.labels) == 0:
            raise ValueError(
                'model.labels is empty. Model entity must have labels')

        params = {'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
                  'train': train_name,
                  'val': val_name,
                  'names': list(self.model_entity.label_to_id_map.keys())
                  }

        data_yaml_filename = os.path.join(data_path, f'{self.model_entity.dataset_id}.yaml')
        yaml_config.update(params)
        with open(data_yaml_filename, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        faas_callback = kwargs.get('on_epoch_end_callback')

        def on_epoch_end(train_obj):

            self.current_epoch = train_obj.epoch
            metrics = train_obj.metrics
            train_obj.plot_metrics()
            if faas_callback is not None:
                faas_callback(self.current_epoch, epochs)
            samples = list()
            NaN_dict = {'box_loss': 1,
                        'cls_loss': 1,
                        'dfl_loss': 1,
                        'mAP50(B)': 0,
                        'mAP50-95(B)': 0,
                        'precision(B)': 0,
                        'recall(B)': 0}
            for metric_name, value in metrics.items():
                legend, figure = metric_name.split('/')
                logger.info(f'Updating figure {figure} with legend {legend} with value {value}')
                if not np.isfinite(value):
                    filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                    filters.add(field='modelId', values=self.model_entity.id)
                    filters.add(field='figure', values=figure)
                    filters.add(field='data.x', values=self.current_epoch - 1)
                    items = self.model_entity.metrics.list(filters=filters)

                    if items.items_count > 0:
                        value = items.items[0].y
                    else:
                        value = NaN_dict.get(figure, 0)
                    logger.warning(f'Value is not finite. For figure {figure} and legend {legend} using value {value}')
                samples.append(dl.PlotSample(figure=figure,
                                             legend=legend,
                                             x=self.current_epoch,
                                             y=value))
            self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)
            # save model output after each epoch end
            self.configuration['start_epoch'] = self.current_epoch + 1
            self.save_to_model(local_path=output_path, cleanup=False)

        self.model.add_callback(event='on_fit_epoch_end', func=on_epoch_end)
        self.model.train(data=data_yaml_filename,
                         exist_ok=True,  # this will override the output dir and will not create a new one
                         resume=resume,
                         epochs=epochs,
                         batch=batch_size,
                         device=device,
                         augment=augment,
                         name=name,
                         workers=0,
                         imgsz=imgsz,
                         project=project_name)
        if 'start_epoch' in self.configuration and self.configuration['start_epoch'] == epochs:
            self.model_entity.configuration['start_epoch'] = 0
            self.model_entity.update()


if __name__ == '__main__':
    model = dl.models.get(model_id='')
    runner = Adapter(model_entity=model)
    item1 = dl.items.get(item_id='')
    runner.predict_items(items=[item1])
