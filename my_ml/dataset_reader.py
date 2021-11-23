from registrable import Registrable


class DatasetReader(Registrable):
    def read(self, file_path: str):
        raise NotImplementedError

    def generate_batches(self, file_path: str, batch_size: int, shuffle: bool):
        raise NotImplementedError
