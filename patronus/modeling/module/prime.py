import collections
import logging
from collections import namedtuple
from functools import cmp_to_key
from typing import Dict, List, Optional, Union

import dateparser as dp
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count

from patronus.etc import Document
from patronus.modeling.mask import IRI, ISFI
from patronus.processing import pipe
from patronus.storing.module import MemoDocStore
from patronus.tooling import stl

Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "content", "meta"])


logger = logging.getLogger(__name__)


class BM25Okapi(ISFI, IRI):
    __slots__ = ("corpus_size", "k1", "b", "store", "b", "avgdl", "epsilon", "idf")

    def __init__(
        self,
        processor,
        index: str = "document",
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        max_cpu: int = None,
        document_store=None,
    ):
        super(BM25Okapi, self).__init__()
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.processor = processor
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.max_cpu = max_cpu if max_cpu is not None else cpu_count() - 1
        self.connector = collections.defaultdict()
        self.store = MemoDocStore(index=index) if not document_store else document_store
        self.tok = collections.defaultdict(list)  # word -> [(idx_1, cnt_1), (idx_2, cnt_2), ..., (idx_k, cnt_k))]

    def index(self, docs: List[Union[str, Dict[str, str]]]):
        nd = self._initialize(docs)
        self._calc_idf(nd)
        self.nd = nd
        # TODO: Parallelize flow

    def _get_all_paragraphs(self) -> List[Paragraph]:
        """
        Split the list of documents in paragraphs
        """
        documents = self.store.get_all_documents()

        paragraphs = []
        p_id = 0
        for doc in documents:
            for p in doc.content.strip().split(
                " "
            ):  # TODO: this assumes paragraphs are separated by "\n\n". Can be switched to paragraph tokenizer.
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(
                    Paragraph(
                        document_id=doc.id,
                        paragraph_id=p_id,
                        content=(p,),
                        meta=doc.meta,
                    )
                )
                p_id += 1
        logger.info(
            "Found %s candidate paragraphs from %s docs in DB",
            len(paragraphs),
            len(documents),
        )
        return paragraphs

    def _initialize(self, corpus):
        # From each document we need token distribution
        nd = {}  # word -> number of documents with word
        num_tok = 0  # number of tokens (words) across all documents
        for idx, document in enumerate(corpus):  # TODO: rewrite using batches instead
            doc = (
                Document.from_dict({"content": document.strip().lower()})
                if isinstance(document, str)
                else Document.from_dict(document)
            )  # add idx -> documnet_id mapping to support retrieving by idx
            self.connector[idx] = doc.id
            self.store.write([doc])
            uni: List[str] = self.processor(doc.content.strip().lower())  # Unified and splitted across atomic units
            self.doc_len.append(len(uni))  # length distribution across corpus
            num_tok += len(uni)  # num of tokens overall including repetitions

            frequencies = {}  # word -> number of occurence(s) within one document
            for token in uni:
                t = token.lower().strip()
                if t not in frequencies:
                    frequencies[t] = 0
                frequencies[t] += 1
            for tok, freq in frequencies.items():
                self.tok[tok].append((idx, freq))

            self.doc_freqs.append(frequencies)  # TODO: test and remove this first document, second, ...

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1  # having occured "s" times within one document it would still count as "1" and not "s".
                except KeyError:
                    nd[word] = 1  # never seen a word.

            self.corpus_size += 1

        self.avgdl = num_tok / self.corpus_size
        return nd

    def _process(self, corpus):
        pool = Pool(self.max_cpu)
        proc_response = pool.map(self.processor, corpus)  # Apply .process(...) for each document
        pool.close()
        pool.join()
        return proc_response

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = np.log(self.corpus_size - freq + 0.5) - np.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def querify(self, query: List[str], doc_ids: Optional[List[int]] = None):
        """Get the score distribution per document(s)"""
        score = np.zeros(self.corpus_size)  # number of documents
        doc_len = np.array(self.doc_len)  # p_1, ..., p_n where n - number of document(s)
        for i, q in enumerate(query):
            # Нужно взять скор всех документов? (for every document the term has occured ... take the product )
            #
            # For every word in query perform the following:
            # fetch the score contribution per document of current word.
            qx = q.lower().strip()

            idf_score = self.idf.get(qx) or 0.0
            # TODO: Here the memory footprint is ZERO until we access specific index. Cudos to Linux!
            #
            # Are you using Linux? Linux has lazy allocation of memory. The underlying calls to malloc and calloc in numpy
            # always 'succeed'.
            # No memory is actually allocated until the memory is first accessed.
            # The zeros function will use calloc which zeros any allocated memory before it is first accessed.
            # Therfore, numpy need not explicitly zero the array and so the array will be lazily initialised.
            # Whereas, the repeat function cannot rely on calloc to initialise the array.
            # Instead it must use malloc and then copy the repeated to all elements in the array
            # (thus forcing immediate allocation).
            tf_score = np.zeros(doc_len.shape)
            for occurence in self.tok[qx]:  # For every document having term `qx`
                pos, token_score = occurence
                tf_score[pos] = token_score

            score += (idf_score) * (
                tf_score * (self.k1 + 1) / (tf_score + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def retrieve_top_k(
        self, query: Union[str, List[str]], topic_ids: List[int] = None, left_date=None, right_date=None, top_k: int = 5
    ):
        """Move the `querify` semantic here to simplify api"""
        # TODO: ... rewrite yielding Generator ...
        left_date = (
            dp.parse(left_date, settings={"DATE_ORDER": "DMY"})
            if left_date is not None and isinstance(left_date, str)
            else left_date
        )
        right_date = (
            dp.parse(right_date, settings={"DATE_ORDER": "DMY"})
            if right_date is not None and isinstance(right_date, str)
            else right_date
        )
        score = self.querify(query)
        ranking = np.argsort(score)[::-1]  # sort in descending order idx - wise
        _ids = [self.connector[i] for i in ranking]
        ranking = ranking[:top_k]
        _ids = _ids[:top_k]
        documents = self.store.get_documents_by_id(ids=_ids)
        documents = sorted(documents, key=cmp_to_key(pipe.pipe_cmp_date))
        it = stl.NIterator(documents)
        response = []
        topic_ids = set(topic_ids) if topic_ids else None
        while it.has_next() and len(response) < top_k:
            doc = it.next()
            cur_timestamp = doc.meta["timestamp"]
            if topic_ids is not None:
                cur_topic_id = doc.meta["topic_id"]
                if cur_topic_id not in topic_ids:
                    continue
            if left_date is not None and cur_timestamp < left_date:
                continue
            if right_date is not None and cur_timestamp > right_date:
                continue
            response.append(doc)

        return response


class BM25L(BM25Okapi):
    __slots__ = ("corpus_size", "k1", "b", "doc_freqs", "b", "avgdl", "delta")

    def __init__(
        self,
        processor,
        index: str = "document",
        k1=1.5,
        delta=0.5,
        b=0.75,
        epsilon=0.25,
        max_cpu: int = None,
    ):
        super(BM25L, self).__init__(processor, index=index, k1=k1, b=b, epsilon=epsilon, max_cpu=max_cpu)

        self.delta = delta

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = np.log(self.corpus_size + 1) - np.log(freq + 0.5)
            self.idf[word] = idf

    def querify(self, query: List[str], doc_ids: List[str] = None):
        """Get the score distribution per document(s)"""
        score = np.zeros(self.corpus_size)  # number of documents
        doc_len = np.array(self.doc_len)  # p_1, ..., p_n where n - number of document(s)
        for i, q in enumerate(query):
            qx = q.lower().strip()

            idf_score = self.idf.get(qx) or 0.0

            tf_score = np.zeros(doc_len.shape)

            for occurence in self.tok[qx]:
                pos, token_score = occurence
                tf_score[pos] = token_score

            # Modify tf_score to take length distribution of the document into account
            tf_score = tf_score / (1 - self.b + self.b * doc_len / self.avgdl)

            # Finally score it
            score += (idf_score) * (self.k1 + 1) * (tf_score + self.delta) / (self.k1 + tf_score + self.delta)

        return score


__all__ = ["BM25Okapi", "BM25L"]
