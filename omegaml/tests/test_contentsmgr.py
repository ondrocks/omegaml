from unittest import TestCase

from tornado.web import HTTPError

from omegaml import Omega
from omegaml.notebook.omegacontentsmgr import OmegaStoreContentsManager


class OmegaContentsManagerTests(TestCase):
    def setUp(self):
        self.om = Omega()
        self.mgr = OmegaStoreContentsManager(omega=self.om)
        [self.om.jobs.drop(fn) for fn in self.om.jobs.list()]

    def _create_notebook(self, name):
        code = """
        print('hello world')
        """.strip()
        self.om.jobs.create(code, name)

    def test_get_top_level(self):
        om = self.om
        # check director is empty to start with
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 0)
        # add a new model, check it is represented
        self._create_notebook('foo')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'foo.ipynb')
        self.assertEqual(model['content'][0]['type'], 'notebook')

    def test_subdirectory(self):
        om = self.om
        # create a notebook in a sub directory
        self._create_notebook('sub/foo')
        model = self.mgr.get('/')
        # check we can see the directory
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['type'], 'directory')
        self.assertEqual(model['content'][0]['name'], 'sub')
        # check we can see the notebook
        model = self.mgr.get('/sub', type='directory')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'foo.ipynb')
        # create a notebook in a sub sub directory
        self._create_notebook('sub1/sub2/foo')
        model = self.mgr.get('/sub1', type='directory')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['type'], 'directory')
        self.assertEqual(model['content'][0]['name'], 'sub2')
        model = self.mgr.get('/sub1/sub2', type='directory')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'foo.ipynb')

    def test_save_notebook(self):
        # create a dummy notebook just so we have a valid model
        self._create_notebook('foo')
        nbmodel = self.mgr.get('foo.ipynb', type='notebook')
        # copy foo to bar
        model = self.mgr._base_model('sub', kind='directory')
        self.mgr.save(model, 'sub')
        model = self.mgr._notebook_model('foo.ipynb')
        self.mgr.save(model, 'sub/bar.ipynb')
        # check it got created within a directory
        model = self.mgr.get('/sub', type='directory')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'bar.ipynb')
        # check it can be read directly
        model = self.mgr.get('/sub/bar.ipynb', type='notebook')
        self.assertEqual(model['name'], 'bar.ipynb')
        self.assertEqual(model['content']['cells'][0]['source'], "print('hello world')")

    def test_overwrite_existing_notebook(self):
        # create a dummy notebook just so we have a valid model
        self._create_notebook('foo')
        nbmodel = self.mgr.get('foo.ipynb', type='notebook')
        # copy foo to bar
        model = self.mgr._base_model('sub', kind='directory')
        self.mgr.save(model, 'sub')
        model = self.mgr._notebook_model('foo.ipynb')
        self.mgr.save(model, 'sub/foo.ipynb')
        # save again
        model['content']['cells'][0]['source'] = "print('hello world at large!')"
        self.mgr.save(model, 'sub/foo.ipynb')
        # check it got created within a directory
        model = self.mgr.get('/sub', type='directory')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'foo.ipynb')
        # check it can be read directly
        model = self.mgr.get('/sub/foo.ipynb', type='notebook')
        self.assertEqual(model['name'], 'foo.ipynb')
        self.assertEqual(model['content']['cells'][0]['source'], "print('hello world at large!')")

    def test_create_directory(self):
        # test creating directory on top level
        model = self.mgr._base_model('sub', kind='directory')
        self.mgr.save(model, 'sub')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'sub')
        self.assertEqual(model['content'][0]['type'], 'directory')
        # test creating directory on sub level
        model = self.mgr._base_model('sub/sub1', kind='directory')
        self.mgr.save(model, 'sub/sub1')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        model = self.mgr.get('/sub', type='directory')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'sub1')
        self.assertEqual(model['content'][0]['type'], 'directory')

    def test_create_directory_escaped(self):
        # test creating directory on top level
        model = self.mgr._base_model('Untitled Folder', kind='directory')
        self.mgr.save(model, 'Untitled Folder')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'Untitled Folder')
        self.assertEqual(model['content'][0]['type'], 'directory')
        # test getting it with escaped string
        self.assertTrue(self.mgr.exists('Untitled Folder'))
        self.assertTrue(self.mgr.exists('Untitled%20Folder'))

    def test_create_notebook_in_empty_directory(self):
        # test creating directory on top level
        model = self.mgr._base_model('sub', kind='directory')
        self.mgr.save(model, 'sub')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'sub')
        self.assertEqual(model['content'][0]['type'], 'directory')
        result = self.mgr.exists('sub')
        self.assertTrue(result)
        # test creating a notebook in this new emtpty directory
        self._create_notebook('sub/foo.ipynb')
        model = self.mgr.get('sub/foo.ipynb', type='notebook')
        self.assertEqual(model['name'], 'foo.ipynb')
        self.assertEqual(model['content']['cells'][0]['source'], "print('hello world')")

    def test_create_multiple_notebook_in_empty_directory(self):
        # test creating directory on top level
        model = self.mgr._base_model('sub', kind='directory')
        self.mgr.save(model, 'sub')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'sub')
        self.assertEqual(model['content'][0]['type'], 'directory')
        result = self.mgr.exists('sub')
        self.assertTrue(result)
        # test creating a notebook in this new emtpty directory
        self._create_notebook('sub/foo.ipynb')
        self._create_notebook('sub/foo2.ipynb')
        self._create_notebook('sub/foo3.ipynb')
        model = self.mgr.get('/')
        self.assertEqual(len(model['content']), 1)
        self.assertEqual(model['content'][0]['name'], 'sub')
        self.assertEqual(model['content'][0]['type'], 'directory')

    def test_exists(self):
        om = self.om
        # check director is empty to start with
        result = self.mgr.exists('/Untitled1')
        self.assertFalse(result)
        result = self.mgr.exists('/sub/sub1')
        self.assertFalse(result)
        # test creating directory on sub level
        model = self.mgr._base_model('sub/sub1', kind='directory')
        self.mgr.save(model, 'sub/sub1')
        result = self.mgr.exists('sub/sub1')
        self.assertTrue(result)
        # test dropping the directory actually works
        self.mgr.delete('sub/sub1')
        result = self.mgr.exists('sub/sub1')
        self.assertFalse(result)
        # test creating a file works
        model = self.mgr._base_model('Notebook.ipynb', kind='notebook')
        result = self.mgr.exists('Notebook.ipynb')

    def test_rename_notebook(self):
        # create a dummy notebook just so we have a valid model
        self._create_notebook('foo')
        nbmodel = self.mgr.get('foo.ipynb', type='notebook')
        self.mgr.rename('foo.ipynb', 'bar.ipynb')

    def test_rename_directory(self):
        model = self.mgr._base_model('sub', kind='directory')
        self.mgr.save(model, 'sub')
        self.mgr.rename_file('sub', 'new_sub')
        with self.assertRaises(HTTPError):
            model = self.mgr.get('sub', type='directory')
        model = self.mgr.get('new_sub', type='directory')
        self.assertEqual(len(model['content']), 0)
