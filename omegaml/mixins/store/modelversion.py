from copy import deepcopy
from hashlib import sha1

# byte string
_u8 = lambda t: t.encode('UTF-8', 'replace') if isinstance(t, str) else t


class ModelVersionMixin(object):
    """
    Versioning support for models

    Usage:
        # create a new version
        om.models.put(model, 'mymodel')

        # get the most recent version
        om.models.get('mymodel', version=-1)

        # tag a model, get specific tag
        om.models.put(model, 'mymodel', tag='foo')
        om.models.get('mymodel', tag='foo')

        # get a specific tag included in name
        om.models.get('mymodel@foo')

        # specify the commit id yourself (e.g. when integrating with git)
        # note this does not interact with git in any way
        om.models.put(model, 'mymodel', commit=<git ref>'

    Notes:
        * every put will create a new version
        * it is not possible to delete a version
        * the versioning is purely based on model Metadata,
          all functionality will continue to work as before
    """

    def put(self, obj, name, tag=None, commit=None, **kwargs):
        # create a new version
        meta = super().put(obj, name, **kwargs)
        if self._model_version_applies():
            self._ensure_versioned(meta)
            meta = self._put_version(meta, tag=tag, commit=commit)
        return meta

    def _model_version_applies(self):
        return self.prefix.startswith('models/')

    def _ensure_versioned(self, meta):
        if 'versions' not in meta.attributes:
            meta.attributes['versions'] = {}
            meta.attributes['versions']['tags'] = {}
            meta.attributes['versions']['commits'] = []
            meta.attributes['versions']['tree'] = {}

    def _put_version(self, meta, tag=None, commit=None):
        version_hash = commit or self._model_version_hash(meta)
        previous = meta.attributes['versions']['tags'].get('latest')
        version_meta = deepcopy(meta)
        version_meta.id = None
        version_meta.name = self._model_version_store_key(meta.name, version_hash)
        del version_meta.attributes['versions']
        version_meta.save()
        meta.attributes['versions']['commits'].append(dict(name=version_meta.name, ref=version_hash))
        meta.attributes['versions']['tree'][version_hash] = previous
        meta.attributes['versions']['tags']['latest'] = version_hash
        if tag:
            meta.attributes['versions']['tags'][tag] = version_hash
        return meta.save()

    def get(self, name, commit=None, tag=None, version=-1, **kwargs):
        actual_name = self._model_version_actual_name(name, tag=tag,
                                                      commit=commit,
                                                      version=version)
        actual_obj = super().get(actual_name, **kwargs)
        return actual_obj

    def xmetadata(self, name, commit=None, tag=None, version=-1, **kwargs):
        meta = super().metadata(name, **kwargs)
        if meta and 'versions' in meta.attributes:
            actual_name = self._model_version_actual_name(name, tag=tag,
                                                          commit=commit,
                                                          version=version)
            meta = super().metadata(actual_name, **kwargs)
        return meta

    def _model_version_actual_name(self, name, tag=None, commit=None,
                                   version=None, **kwargs):
        if '@' in name:
            name, tag = name.split('@')
            commit = tag
        meta = super().metadata(name, **kwargs)
        if meta is None:
            return name
        if tag or commit:
            if tag and tag in meta.attributes['versions']['tags']:
                version_hash = meta.attributes['versions']['tags'][tag]
                actual_name = self._model_version_store_key(meta.name, version_hash)
            elif commit:
                actual_name = self._model_version_store_key(meta.name, commit)
            else:
                actual_name = name
        else:
            if abs(version) <= len(meta.attributes['versions']['commits']):
                actual_name = meta.attributes['versions']['commits'][version]['name']
            else:
                actual_name = name
        return actual_name

    def _model_version_hash(self, meta):
        hasher = sha1()
        hasher.update(_u8(meta.name))
        hasher.update(_u8(str(meta.modified)))
        return hasher.hexdigest()

    def _model_version_store_key(self, name, version_hash):
        return '_versions/{}/{}'.format(name, version_hash)
