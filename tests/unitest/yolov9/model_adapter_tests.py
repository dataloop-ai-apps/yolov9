import unittest
import dtlpy as dl
import os
import json
import enum


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
ENV = os.environ['ENV']
DATASET_NAME = "YoloV9-Tests"


class ItemTypes(enum.Enum):
    IMAGE = "image"


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    assets_folder: str = os.path.join('tests', 'assets', 'unittest')
    dataset_folder: str = os.path.join(assets_folder, 'datasets', 'example_data')
    prepare_item_function = dict()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv(ENV)
        dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)
        cls.prepare_item_function = {
            ItemTypes.IMAGE.value: cls._prepare_image_item
        }

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all models
        for model in cls.project.models.list().all():
            model.delete()

        # Delete all apps
        for app in cls.project.apps.list().all():
            if app.project.id == cls.project.id:
                app.uninstall()

        # Delete all dpks
        filters = dl.Filters(resource=dl.FiltersResource.DPK)
        filters.add(field="scope", values="project")
        for dpk in cls.project.dpks.list(filters=filters).all():
            if dpk.project.id == cls.project.id and dpk.creator == BOT_EMAIL:
                dpk.delete()
        dl.logout()

    # Item preparation functions
    def _prepare_image_item(self, item_name: str):
        local_path = os.path.join(self.dataset_folder, item_name)
        item = self.dataset.items.upload(
            local_path=local_path,
            overwrite=True
        )
        return item

    # Perdict function
    def _perform_model_predict(self, item_type: ItemTypes, item_name: str, model_index: int):
        # Upload item
        item = self.prepare_item_function[item_type.value](self=self, item_name=item_name)

        # Open dataloop json
        dataloop_json_filepath = 'dataloop.json'
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        model_name = dataloop_json.get('components', dict()).get('models', list())[model_index].get("name", None)

        try:
            model = self.project.models.get(model_name=model_name)
        except dl.exceptions.NotFound:
            # Publish dpk and install app
            dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
            dpk = self.project.dpks.publish(dpk=dpk)
            app = self.project.apps.install(dpk=dpk)

            # Get model and predict
            model = app.project.models.get(model_name=model_name)

        service = model.deploy()
        model.metadata["system"]["deploy"] = {"services": [service.id]}
        execution = model.predict(item_ids=[item.id])
        execution = execution.wait()

        # Execution output format:
        # [[{"item_id": item_id}, ...], [{"annotation_id": annotation_id}, ...]]
        _, annotations = execution.output
        return annotations

    # Test functions
    def test_yolov9c(self):
        item_name = "car_image.jpeg"
        item_type = ItemTypes.IMAGE
        predicted_annotations = self._perform_model_predict(item_type=item_type, item_name=item_name, model_index=0)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)
        car_annotation = self.dataset.annotations.get(annotation_id=predicted_annotations[0]["annotation_id"])
        self.assertTrue(car_annotation.type == dl.AnnotationType.BOX and car_annotation.label == "car")

    def test_yolov9e(self):
        item_name = "car_image.jpeg"
        item_type = ItemTypes.IMAGE
        predicted_annotations = self._perform_model_predict(item_type=item_type, item_name=item_name, model_index=1)
        self.assertTrue(isinstance(predicted_annotations, list) and len(predicted_annotations) > 0)
        car_annotation = self.dataset.annotations.get(annotation_id=predicted_annotations[0]["annotation_id"])
        self.assertTrue(car_annotation.type == dl.AnnotationType.BOX and car_annotation.label == "car")


if __name__ == '__main__':
    unittest.main()
