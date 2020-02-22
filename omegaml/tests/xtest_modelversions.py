from unittest import TestCase

from sklearn.linear_model import LinearRegression

from omegaml import Omega
from omegaml.mixins.store.modelversion import ModelVersionMixin
from omegaml.tests.util import OmegaTestMixin


class ModelVersionMixinTests(OmegaTestMixin, TestCase):
    def setUp(self):
        self.om = Omega()
        self.clean()

    def test_version_on_put(self):
        store = self.om.models
        store.register_mixin(ModelVersionMixin)
        clf = LinearRegression()
        meta = store.put(clf, 'regmodel')
        self.assertIn('versions', meta.attributes)
        models = store.list(include_temp=True)
        latest = meta.attributes['versions']['tags']['latest']
        store_key = store._model_version_store_key('regmodel', latest)
        self.assertIn(store_key, models)

    def test_get_version_by_index(self):
        store = self.om.models
        store.register_mixin(ModelVersionMixin)
        clf = LinearRegression()
        clf.version_ = 1
        meta = store.put(clf, 'regmodel')
        clf.version_ = 2
        meta = store.put(clf, 'regmodel')
        clf_ = store.get('regmodel', version=-1)
        self.assertEqual(clf_.version_, 2)
        clf_ = store.get('regmodel', version=-2)
        self.assertEqual(clf_.version_, 1)
        clf_ = store.get('regmodel')
        self.assertEqual(clf_.version_, 2)

    def test_get_version_by_tag(self):
        store = self.om.models
        store.register_mixin(ModelVersionMixin)
        clf = LinearRegression()
        clf.version_ = 1
        meta = store.put(clf, 'regmodel', tag='version1')
        clf.version_ = 2
        meta = store.put(clf, 'regmodel', tag='version2')
        clf_ = store.get('regmodel', tag='version2')
        self.assertEqual(clf_.version_, 2)
        clf_ = store.get('regmodel', tag='version1')
        self.assertEqual(clf_.version_, 1)
        clf_ = store.get('regmodel')
        self.assertEqual(clf_.version_, 2)

    def test_get_version_by_hashtag(self):
        store = self.om.models
        store.register_mixin(ModelVersionMixin)
        clf = LinearRegression()
        clf.version_ = 1
        meta = store.put(clf, 'regmodel', tag='version1')
        clf.version_ = 2
        meta = store.put(clf, 'regmodel', tag='version2')
        clf_ = store.get('regmodel@version2')
        self.assertEqual(clf_.version_, 2)
        clf_ = store.get('regmodel@version1')
        self.assertEqual(clf_.version_, 1)
        clf_ = store.get('regmodel')
        self.assertEqual(clf_.version_, 2)

    def test_get_version_by_commit(self):
        store = self.om.models
        store.register_mixin(ModelVersionMixin)
        clf = LinearRegression()
        clf.version_ = 1
        meta = store.put(clf, 'regmodel')
        clf.version_ = 2
        meta = store.put(clf, 'regmodel')
        meta = store.metadata('regmodel')
        commit1 = meta.attributes['versions']['commits'][-2]['ref']
        commit2 = meta.attributes['versions']['commits'][-1]['ref']
        clf_ = store.get('regmodel', commit=commit2)
        self.assertEqual(clf_.version_, 2)
        clf_ = store.get('regmodel', commit=commit1)
        self.assertEqual(clf_.version_, 1)
        clf_ = store.get('regmodel')
        self.assertEqual(clf_.version_, 2)

    def test_get_metadata_by_version(self):
        store = self.om.models
        store.register_mixin(ModelVersionMixin)
        clf = LinearRegression()
        clf.version_ = 1
        store.put(clf, 'regmodel', tag='commit1')
        clf.version_ = 2
        meta = store.put(clf, 'regmodel', tag='commit2')
        meta_commit1 = store.metadata(meta.attributes['versions']['commits'][-2]['name'])
        meta_commit2= store.metadata(meta.attributes['versions']['commits'][-1]['name'])
        meta_ = store.metadata('regmodel@commit1')
        self.assertEqual(meta_.id, meta_commit1.id)
        meta_ = store.metadata('regmodel@commit2')
        self.assertEqual(meta_.id, meta_commit2.id)
