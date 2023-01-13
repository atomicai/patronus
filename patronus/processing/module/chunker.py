from nnsplit import NNSplit

from patronus.processing.mask import IChunker, IPath


class Chunker(IChunker):
    def __init__(self, num_tokens: int = 256, window_size: int = 64):
        model = NNSplit(model_path=IPath.runnsplitpath)
        super(Chunker, self).__init__(model, num_tokens=num_tokens, window_size=window_size)

    def chunkify(self, text: str, num_tokens: int = None, window_size: int = None, as_document: bool = False):
        num_tokens = num_tokens or self.num_tokens
        window_size = window_size or self.window_size
        response = []
        for sentence in self.model.split([text])[0]:
            response.append(str(sentence))
        return response


__all__ = ["Chunker"]
