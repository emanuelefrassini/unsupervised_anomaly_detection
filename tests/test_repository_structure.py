"""Standard-library, data-free structure checks; NOT ML performance tests."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryStructureTests(unittest.TestCase):
    def test_research_script_is_syntactically_valid(self):
        path = ROOT / 'VAE_reproducible_improved.py'
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_research_model_contract(self):
        tree = ast.parse((ROOT / 'VAE_reproducible_improved.py').read_text(encoding='utf-8'))
        declared = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        self.assertTrue({'VAE', 'MinMaxScaler', 'vae_loss_function',
                         'find_optimal_threshold_f1', 'evaluate_performance'} <= declared)

    def test_paper_citation_and_scope(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
        self.assertIn('10.1080/13645706.2026.2705585', readme)
        self.assertIn('10.1080/13645706.2026.2705585', citation)
        self.assertIn('not', readme.lower())
        self.assertIn('synthetic', readme.lower())

    def test_auxiliary_python_syntax(self):
        for name in ('concatenate_pose_rgb.py', 'convert_all_json_to_pyskl_10sclip.py',
                     'create_conf_files.py', 'create_conf_files_videos.py',
                     'process_videos_optimized_2.py', 'test.py', 'videos_chunks.py'):
            with self.subTest(name=name):
                path = ROOT / name
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


if __name__ == '__main__':
    unittest.main()
