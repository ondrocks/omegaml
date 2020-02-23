from __future__ import absolute_import

import glob
import os
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import datetime
import joblib
from mongoengine.fields import GridFSProxy
from shutil import rmtree
from sklearn.model_selection import GridSearchCV

from omegaml.backends.basemodel import BaseModelBackend
from omegaml.documents import MDREGISTRY
from omegaml.util import reshaped, gsreshaped

# byte string
_u8 = lambda t: t.encode('UTF-8', 'replace') if isinstance(t, str) else t


class ScikitLearnBackendV1(BaseModelBackend):
    # kept to support legacy scikit learn model serializations prior to ~scikit learn v0.18
    def _v1_package_model(self, model, filename):
        """
        Dumps a model using joblib and packages all of joblib files into a zip
        file
        """
        import joblib
        lpath = tempfile.mkdtemp()
        fname = os.path.basename(filename)
        mklfname = os.path.join(lpath, fname)
        zipfname = os.path.join(self.model_store.tmppath, fname)
        joblib.dump(model, mklfname, protocol=4)
        with ZipFile(zipfname, 'w', compression=ZIP_DEFLATED) as zipf:
            for part in glob.glob(os.path.join(lpath, '*')):
                zipf.write(part, os.path.basename(part))
        rmtree(lpath)
        return zipfname

    def _v1_extract_model(self, packagefname):
        """
        Loads a model using joblib from a zip file created with _package_model
        """
        import joblib
        lpath = tempfile.mkdtemp()
        fname = os.path.basename(packagefname)
        mklfname = os.path.join(lpath, fname)
        with ZipFile(packagefname) as zipf:
            zipf.extractall(lpath)
        model = joblib.load(mklfname)
        rmtree(lpath)
        return model

    def get_model(self, name, version=-1):
        """
        Retrieves a pre-stored model
        """
        filename = self.model_store._get_obj_store_key(name, '.omm')
        packagefname = os.path.join(self.model_store.tmppath, name)
        dirname = os.path.dirname(packagefname)
        try:
            os.makedirs(dirname)
        except OSError:
            # OSError is raised if path exists already
            pass
        outf = self.model_store.fs.get_version(filename, version=version)
        with open(packagefname, 'wb') as zipf:
            zipf.write(outf.read())
        model = self._v1_extract_model(packagefname)
        return model

    def put_model(self, obj, name, attributes=None):
        """
        Packages a model using joblib and stores in GridFS
        """
        zipfname = self._v1_package_model(obj, name)
        with open(zipfname, 'rb') as fzip:
            fileid = self.model_store.fs.put(
                fzip, filename=self.model_store._get_obj_store_key(name, 'omm'))
            gridfile = GridFSProxy(grid_id=fileid,
                                   db_alias='omega',
                                   collection_name=self.model_store.bucket)
        return self.model_store._make_metadata(
            name=name,
            prefix=self.model_store.prefix,
            bucket=self.model_store.bucket,
            kind=MDREGISTRY.SKLEARN_JOBLIB,
            attributes=attributes,
            gridfile=gridfile).save()


class ScikitLearnBackendV2(ScikitLearnBackendV1):
    """
    OmegaML backend to use with ScikitLearn
    """
    _backend_version_tag = '_om_backend_version'
    _backend_version = '2'

    def _v2_package_model(self, model, key):
        """
        Dumps a model using joblib and packages all of joblib files into a zip
        file
        """
        packagefname = self._tmp_packagefn(key)
        joblib.dump(model, packagefname, protocol=4)
        return packagefname

    def _v2_extract_model(self, infile, key):
        """
        Loads a model using joblib from a zip file created with _package_model
        """
        packagefname = self._tmp_packagefn(key)
        with open(packagefname, 'wb') as pkgf:
            pkgf.write(infile.read())
        model = joblib.load(packagefname)
        return model

    def _tmp_packagefn(self, name):
        filename = os.path.join(self.model_store.tmppath, name)
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        return filename

    def get_model(self, name, version=-1):
        """
        Retrieves a pre-stored model
        """

        meta = self.model_store.metadata(name)
        if self._backend_version_tag not in meta.kind_meta:
            return super().get_model(name, version=version)
        storekey = self.model_store.object_store_key(name, 'omm', hashed=True)
        model = self._v2_extract_model(meta.gridfile, storekey)
        return model

    def put_model(self, obj, name, attributes=None, _kind_version=None):
        """
        Packages a model using joblib and stores in GridFS
        """
        if _kind_version and _kind_version != self._backend_version:
            return super().put_model(obj, name, attributes=attributes)
        storekey = self.model_store.object_store_key(name, 'omm', hashed=True)
        packagefname = self._v2_package_model(obj, storekey)
        gridfile = GridFSProxy(db_alias='omega',
                               collection_name=self.model_store._fs_collection)
        with open(packagefname, 'rb') as pkgf:
            fileid = gridfile.put(pkgf, filename=storekey)
        kind_meta = {
            self._backend_version_tag: self._backend_version,
        }
        return self.model_store._make_metadata(
            name=name,
            prefix=self.model_store.prefix,
            bucket=self.model_store.bucket,
            kind=MDREGISTRY.SKLEARN_JOBLIB,
            kind_meta=kind_meta,
            attributes=attributes,
            gridfile=gridfile).save()

    def predict(
          self, modelname, Xname, rName=None, pure_python=True, **kwargs):
        data = self.data_store.get(Xname)
        model = self.model_store.get(modelname)
        result = model.predict(reshaped(data), **kwargs)
        if pure_python:
            result = result.tolist()
        if rName:
            meta = self.data_store.put(result, rName)
            result = meta
        return result

    def predict_proba(
          self, modelname, Xname, rName=None, pure_python=True, **kwargs):
        data = self.data_store.get(Xname)
        model = self.model_store.get(modelname)
        result = model.predict_proba(reshaped(data), **kwargs)
        if pure_python:
            result = result.tolist()
        if rName:
            meta = self.data_store.put(result, rName)
            result = meta
        return result

    def fit(self, modelname, Xname, Yname=None, pure_python=True, **kwargs):
        model = self.model_store.get(modelname)
        X, metaX = self.data_store.get(Xname), self.data_store.metadata(Xname)
        Y, metaY = None, None
        if Yname:
            Y, metaY = (self.data_store.get(Yname),
                        self.data_store.metadata(Yname))
        model.fit(reshaped(X), reshaped(Y), **kwargs)
        # store information required for retraining
        model_attrs = {
            'metaX': metaX.to_mongo(),
            'metaY': metaY.to_mongo() if metaY is not None else None,
        }
        try:
            import sklearn
            model_attrs['scikit-learn'] = sklearn.__version__
        except:
            model_attrs['scikit-learn'] = 'unknown'
        meta = self.model_store.put(model, modelname, attributes=model_attrs)
        return meta

    def partial_fit(
          self, modelname, Xname, Yname=None, pure_python=True, **kwargs):
        model = self.model_store.get(modelname)
        X, metaX = self.data_store.get(Xname), self.data_store.metadata(Xname)
        Y, metaY = None, None
        if Yname:
            Y, metaY = (self.data_store.get(Yname),
                        self.data_store.metadata(Yname))
        model.partial_fit(reshaped(X), reshaped(Y), **kwargs)
        # store information required for retraining
        model_attrs = {
            'metaX': metaX.to_mongo(),
            'metaY': metaY.to_mongo() if metaY is not None else None,
        }
        try:
            import sklearn
            model_attrs['scikit-learn'] = sklearn.__version__
        except:
            model_attrs['scikit-learn'] = 'unknown'
        meta = self.model_store.put(model, modelname, attributes=model_attrs)
        return meta

    def score(
          self, modelname, Xname, Yname, rName=None, pure_python=True,
          **kwargs):
        model = self.model_store.get(modelname)
        X = self.data_store.get(Xname)
        Y = self.data_store.get(Yname)
        result = model.score(reshaped(X), reshaped(Y), **kwargs)
        if rName:
            meta = self.model_store.put(result, rName)
            result = meta
        return result

    def fit_transform(
          self, modelname, Xname, Yname=None, rName=None, pure_python=True,
          **kwargs):
        model = self.model_store.get(modelname)
        X, metaX = self.data_store.get(Xname), self.data_store.metadata(Xname)
        Y, metaY = None, None
        if Yname:
            Y, metaY = (self.data_store.get(Yname),
                        self.data_store.metadata(Yname))
        result = model.fit_transform(reshaped(X), reshaped(Y), **kwargs)
        # store information required for retraining
        model_attrs = {
            'metaX': metaX.to_mongo(),
            'metaY': metaY.to_mongo() if metaY is not None else None
        }
        try:
            import sklearn
            model_attrs['scikit-learn'] = sklearn.__version__
        except:
            model_attrs['scikit-learn'] = 'unknown'
        meta = self.model_store.put(model, modelname, attributes=model_attrs)
        if pure_python:
            result = result.tolist()
        if rName:
            meta = self.data_store.put(result, rName)
        result = meta
        return result

    def transform(self, modelname, Xname, rName=None, pure_python=True, **kwargs):
        model = self.model_store.get(modelname)
        X = self.data_store.get(Xname)
        result = model.transform(reshaped(X), **kwargs)
        if pure_python:
            result = result.tolist()
        if rName:
            meta = self.data_store.put(result, rName)
            result = meta
        return result

    def decision_function(self, modelname, Xname, rName=None, pure_python=True, **kwargs):
        model = self.model_store.get(modelname)
        X = self.data_store.get(Xname)
        result = model.decision_function(reshaped(X), **kwargs)
        if pure_python:
            result = result.tolist()
        if rName:
            meta = self.data_store.put(result, rName)
            result = meta
        return result

    def gridsearch(self, modelname, Xname, Yname, rName=None,
                   parameters=None, pure_python=True, **kwargs):
        model, meta = self.model_store.get(modelname), self.model_store.metadata(modelname)
        X = self.data_store.get(Xname)
        if Yname:
            y = self.data_store.get(Yname)
        else:
            y = None
        gs_model = GridSearchCV(cv=5, estimator=model, param_grid=parameters, **kwargs)
        gs_model.fit(X, gsreshaped(y))
        nowdt = datetime.datetime.now()
        if rName:
            gs_modelname = rName
        else:
            gs_modelname = '{}.{}.gs'.format(modelname, nowdt.isoformat())
        gs_meta = self.model_store.put(gs_model, gs_modelname)
        attributes = meta.attributes
        if not 'gridsearch' in attributes:
            attributes['gridsearch'] = []
        attributes['gridsearch'].append({
            'datetime': nowdt,
            'Xname': Xname,
            'Yname': Yname,
            'gsModel': gs_modelname,
        })
        meta.save()
        return meta


class ScikitLearnBackend(ScikitLearnBackendV2):
    pass
