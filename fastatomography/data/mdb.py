from builtins import property

from fastatomography.util.parameters import Param


class MDBEntry:
    projections_url = property()
    volume_url = property()
    xyz_url = property()
    DOI = property()
    title = property()
    atomic_species = property()

    def __init__(self, projections_url, volume_url, xyz_url, DOI, title='', species=[]):
        self.projections_url = projections_url
        self.volume_url = volume_url
        self.xyz_url = xyz_url
        self.DOI = DOI
        self.title = title
        self.atomic_species = species

projections_base_url = 'https://www.materialsdatabank.org/api/v1/mdb/datasets/_/projections/'
reconstructions_base_url = 'https://www.materialsdatabank.org/api/v1/mdb/datasets/_/reconstructions/'
structures_base_url = 'https://www.materialsdatabank.org/api/v1/mdb/datasets/_/structures/'

Xu2015 = MDBEntry(
    projections_url=projections_base_url + '5d047d146f07c20001127a03/emd?updated=2020-01-28T02:06:15.345000+00:00',
    volume_url=reconstructions_base_url + '5d047d146f07c20001127a02/emd?updated=2020-01-28T02:06:15.345000+00:00',
    xyz_url=structures_base_url + '5d047d136f07c20001127a01/xyz?updated=2020-01-28T02:06:15.345000+00:00',
    DOI='10.1038/nmat4426',
    title='Three-dimensional coordinates of individual atoms in materials revealed by electron tomography',
    species=[74])

Tian2020 = MDBEntry(
    projections_url=projections_base_url + '5e66830e4fdc3e0001a7e4ee/emd?updated=2020-03-09T17:56:16.182000+00:00',
    volume_url=reconstructions_base_url + '5e66830e4fdc3e0001a7e4ed/emd?updated=2020-03-09T17:56:16.182000+00:00',
    xyz_url=structures_base_url + '5e66830d4fdc3e0001a7e4ec/xyz?updated=2020-03-09T17:56:16.182000+00:00',
    DOI='10.1038/s41563-020-0636-5',
    title='Correlating the three-dimensional atomic defects and electronic properties of two-dimensional transition metal dichalcogenides',
    species=[16, 42, 75])

Yang2017 = MDBEntry(
    projections_url=projections_base_url + '5d0480066f07c20001127a1d/emd?updated=2020-01-28T02:06:54.122000+00:00',
    volume_url=reconstructions_base_url + '5d0480066f07c20001127a1c/emd?updated=2020-01-28T02:06:54.122000+00:00',
    xyz_url=structures_base_url + '5d0480056f07c20001127a1b/xyz?updated=2020-01-28T02:06:54.122000+00:00',
    DOI='10.1038/nature21042',
    title='Deciphering chemical order/disorder and material properties at the single-atom level',
    species=[26, 78])

Zhou2019_1_1 = MDBEntry(
    projections_url=projections_base_url + '5d0480976f07c20001127a36/emd?updated=2020-01-28T02:10:40.573000+00:00',
    volume_url=reconstructions_base_url + '5d0480966f07c20001127a35/emd?updated=2020-01-28T02:10:40.573000+00:00',
    xyz_url=structures_base_url + '5d0480966f07c20001127a34/xyz?updated=2020-01-28T02:10:40.573000+00:00',
    DOI='10.1038/s41586-019-1317-x',
    title='Observing crystal nucleation in four dimensions using atomic electron tomography',
    species=[26, 78])

Zhou2019_1_2 = MDBEntry(
    projections_url=projections_base_url + '5d0481796f07c20001127a5a/emd?updated=2020-01-28T02:10:55.091000+00:00',
    volume_url=reconstructions_base_url + '5d0481796f07c20001127a59/emd?updated=2020-01-28T02:10:55.091000+00:00',
    xyz_url=structures_base_url + '5d0481786f07c20001127a58/xyz?updated=2020-01-28T02:10:55.091000+00:00',
    DOI='10.1038/s41586-019-1317-x',
    title='Observing crystal nucleation in four dimensions using atomic electron tomography',
    species=[26, 78])
















