from __future__ import absolute_import

import glob
import os
import tempfile
from shutil import rmtree
from zipfile import ZipFile, ZIP_DEFLATED

import datetime
import joblib
from mongoengine.fields import GridFSProxy
from sklearn.model_selection import GridSearchCV

from omegaml.backends.basemodel import BaseModelBackend
from omegaml.documents import MDREGISTRY
from omegaml.util import reshaped, gsreshaped
from hashlib import sha1

# byte string
_u8 = lambda t: t.encode('UTF-8', 'replace') if isinstance(t, str) else t


class ScikitLearnBackend(BaseModelBackend):
    """
    OmegaML backend to use with ScikitLearn
    """

    def _package_model(self, model, filename):
        """
        Dumps a model using joblib and packages all of joblib files into a zip
        file
        """
        joblib.dump(model, filename, protocol=4)
        return filename

    def _extract_model(self, packagefname):
        """
        Loads a model using joblib from a zip file created with _package_model
        """
        model = joblib.load(packagefname)
        return model

    def get_model(self, name, version=-1):
        """
        Retrieves a pre-stored model
        """
        meta = self.model_store.metadata(name)
        outf = meta.gridfile
        packagefname = os.path.join(self.model_store.tmppath, outf.name + '.omm')
        dirname = os.path.dirname(packagefname)
        os.makedirs(dirname, exist_ok=True)
        with open(packagefname, 'wb') as zipf:
            zipf.write(outf.read())
        model = self._extract_model(packagefname)
        return model

    def put_model(self, obj, name, attributes=None):
        """
        Packages a model using joblib and stores in GridFS
        """
        hasher = sha1()
        hasher.update(_u8(self.model_store._get_obj_store_key(name, 'omm')))
        filename = hasher.hexdigest()
        zipfname = self._package_model(obj, filename)
        with open(zipfname, 'rb') as fzip:
            fileid = self.model_store.fs.put(
                fzip, filename=filename)
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
