from __future__ import unicode_literals

import logging
import os

from opensfm import dataset
from opensfm import transformations as tf
from opensfm import io

logger = logging.getLogger(__name__)


class Command:
    name = 'export_visualsfm'
    help = "Export reconstruction to NVM_V3 format from VisualSfM"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')
        parser.add_argument('--undistorted',
                            action='store_true',
                            help='export the undistorted reconstruction')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        if args.undistorted:
            reconstructions = data.load_undistorted_reconstruction()
            graph = data.load_undistorted_tracks_graph()
        else:
            reconstructions = data.load_reconstruction()
            graph = data.load_tracks_graph()

        if reconstructions:
            self.export(reconstructions[0], graph, data)

    def export(self, reconstruction, graph, data):
        lines = ['NVM_V3', '', str(len(reconstruction.shots))]
        shots = reconstruction.shots.values()
        for shot in shots:
            q = tf.quaternion_from_matrix(shot.pose.get_rotation_matrix())
            o = shot.pose.get_origin()
            words = [
                self.image_path(shot.id, data),
                shot.camera.focal * max(shot.camera.width, shot.camera.height),
                q[0], q[1], q[2], q[3],
                o[0], o[1], o[2],
                '0', '0',
            ]
            lines.append(' '.join(map(str, words)))
        shot_ids_to_indices = { shot.id : i for i, shot in enumerate(shots) }
        lines += [str(len(reconstruction.points))]
        for point in reconstruction.points.values():
            p, c = point.coordinates, point.color
            fragments = [p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2]), len(graph[point.id])]
            for shot_id in graph[point.id]:
                exif = data.load_exif(shot_id)
                try:
                    view_id = shot_ids_to_indices[shot_id]
                except:
                    continue
                feature_id = graph[shot_id][point.id]['feature_id']
                feature_x, feature_y = graph[shot_id][point.id]['feature']
                feature_x, feature_y = exif['width'] * (0.5 + feature_x), exif['height'] * (0.5 + feature_y)
                fragments.extend([view_id, feature_id, int(round(feature_x)), int(round(feature_y))])
            lines.append(' '.join(map(str, fragments)))
        lines += ['0']

        with io.open_wt(data.data_path + '/reconstruction.nvm') as fout:
            fout.write('\n'.join(lines))

    def image_path(self, image, data):
        """Path to the undistorted image relative to the dataset path."""
        path = data._undistorted_image_file(image)
        return os.path.relpath(path, data.data_path)
