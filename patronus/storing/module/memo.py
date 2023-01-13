import collections
import logging
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch

from patronus.etc import Document, DuplicateDocumentError
from patronus.processing.module.filter import LogicalFilterClause
from patronus.storing.mask import BaseDocStore
from patronus.tooling import initialize_device_settings

logger = logging.getLogger(__name__)


class MemoDocStore(BaseDocStore):
    def __init__(
        self,
        index: str = "document",
        embedding_field: Optional[str] = "embedding",
        embedding_dim: int = 768,
        return_embedding: bool = False,
        similarity: str = "dot_product",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        use_gpu: bool = True,
        scoring_batch_size: int = 10_000,
        device: str = "cpu",
    ):
        super(MemoDocStore, self).__init__()
        self.indexes: Dict[str, Dict] = collections.defaultdict(dict)
        self.index: str = index
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.similarity = similarity
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents
        self.use_gpu = use_gpu
        self.scoring_batch_size = scoring_batch_size

        self.replica = ("skip", "overwrite", "fail")
        self.duplicate_documents = duplicate_documents
        self.devices, _ = initialize_device_settings(devices=[device], use_cuda=self.use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        self.main_device = self.devices[0]

    def write(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
         Indexes documents for later queries.
        :param documents: a list of Python dictionaries or a list of  /etc/Document objects.
                           For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                           Optionally: Include meta data via {"content": "<the-actual-text>",
                           "meta": {"name": "<some-document-name>, "author": "somebody", ...}}
                           It can be used for filtering and is accessible in the responses of the Finder.
         :param index: write documents to a custom namespace. For instance, documents for evaluation can be indexed in a
                       separate index than the documents for search.
         :param duplicate_documents: Handle duplicates document based on parameter options.
                                     Parameter options : ( 'skip','overwrite','fail')
                                     skip: Ignore the duplicates documents
                                     overwrite: Update any existing documents with the same ID when adding documents.
                                     fail: an error is raised if the document ID of the document being added already
                                     exists.
         :raises DuplicateDocumentError: Exception trigger on duplicate document
         :return: None
        """

        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.replica, f"duplicate_documents parameter must be {', '.join(self.replica)}"

        field_map = self._create_document_field_map()
        documents = deepcopy(documents)
        documents_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        documents_objects = self._drop_duplicate_documents(documents=documents_objects)
        for document in documents_objects:
            if document.id in self.indexes[index]:
                if duplicate_documents == "fail":
                    raise DuplicateDocumentError(f"Document with id '{document.id} already " f"exists in index '{index}'")
                if duplicate_documents == "skip":
                    logger.warning(f"Duplicate Documents: Document with id '{document.id} already exists in index " f"'{index}'")
                    continue
            self.indexes[index][document.id] = document

    def _create_document_field_map(self):
        return {self.embedding_field: "embedding"}

    def get_document_by_id(
        self, _id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """
        Fetch a document by specifying its text id string.
        """
        if headers:
            raise NotImplementedError("MemoDocStore does not support headers.")

        index = index or self.index
        documents = self.get_documents_by_id([_id], index=index)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:  # type: ignore
        """
        Fetch documents by specifying a list of text id strings.
        """
        index = index or self.index
        documents = [self.indexes[index][_id] for _id in ids]
        return documents

    def _drop_duplicate_documents(self, documents: List[Document], index: Optional[str] = None) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID
        :param documents: A list of Document objects.
        :param index: name of the index
        :return: A list Document objects.
        """
        _hash_ids = set([])
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    f"Duplicate Documents: Document with id '{document.id}' already exists in index " f"'{index or self.index}'"
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents

    def scorify(self, query_vec: np.ndarray, search_space: List[Document]) -> List[float]:
        """
        Calculate similarity scores between query embedding and a list of documents using torch.
        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param document_to_search: List of documents to compare `query_emb` against.
        """
        query_vec = torch.tensor(query_vec, dtype=torch.float).to(self.main_device)

        if len(query_vec.shape) == 1:
            query_vec = query_vec.unsqueeze(dim=0)

        doc_embeds = np.array([doc.embedding for doc in search_space])
        doc_embeds = torch.as_tensor(doc_embeds, dtype=torch.float)
        if len(doc_embeds.shape) == 1 and doc_embeds.shape[0] == 1:
            doc_embeds = doc_embeds.unsqueeze(dim=0)
        elif len(doc_embeds.shape) == 1 and doc_embeds.shape[0] == 0:
            return []

        if self.similarity == "cosine":
            # cosine similarity is just a normed dot product
            query_vec_norm = torch.norm(query_vec, dim=1)
            query_vec = torch.div(query_vec, query_vec_norm)

            doc_vecs_norms = torch.norm(doc_embeds, dim=1)
            doc_vecs = torch.div(doc_embeds.T, doc_vecs_norms).T

        curr_pos = 0
        scores = []
        while curr_pos < len(doc_embeds):
            doc_embeds_slice = doc_vecs[curr_pos : curr_pos + self.scoring_batch_size]
            doc_embeds_slice = doc_embeds_slice.to(self.main_device)
            with torch.no_grad():
                slice_scores = torch.matmul(doc_embeds_slice, query_vec.T).cpu()
                slice_scores = slice_scores.squeeze(dim=1)
                slice_scores = slice_scores.numpy().tolist()

            scores.extend(slice_scores)
            curr_pos += self.scoring_batch_size

        return scores

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get all documents from the document store as a list.
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param return_embedding: Whether to return the document embeddings.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def spacify(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in MemoDocStore
        return_embedding: Optional[bool] = None,
        only_documents_without_embedding: bool = False,
    ):
        index = index or self.index
        documents = deepcopy(list(self.indexes[index].values()))
        documents = [d for d in documents if isinstance(d, Document)]

        if return_embedding is None:
            return_embedding = self.return_embedding
        if return_embedding is False:
            for doc in documents:
                doc.embedding = None

        if only_documents_without_embedding:
            documents = [doc for doc in documents if doc.embedding is None]
        if filters:
            parsed_filter = LogicalFilterClause.parse(filters)
            filtered_documents = list(filter(lambda doc: parsed_filter.evaluate(doc.meta), documents))
        else:
            filtered_documents = documents

        return filtered_documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. The methods returns a Python Generator that yields individual
        documents.
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param return_embedding: Whether to return the document embeddings.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        response = self.spacify(index=index, filters=filters, return_embedding=return_embedding)
        yield from response


__all__ = ["MemoDocStore"]
