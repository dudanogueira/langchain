"""Test Weaviate Multitenant functionality."""
import logging
import os
import uuid
from typing import Generator, Union

import pytest

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests/vectorstores/docker-compose
docker compose -f weaviate.yml up
"""


class TestWeaviate:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    @pytest.fixture(scope="class", autouse=True)
    def weaviate_url(self) -> Union[str, Generator[str, None, None]]:
        """Return the weaviate url."""
        from weaviate import Client

        url = "http://localhost:8080"
        yield url

        # Clear the test index
        client = Client(url)
        client.schema.delete_all()

    @pytest.mark.vcr(ignore_localhost=True)
    def test_multitenant_similarity_search_without_metadata(
        self, weaviate_url: str, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search without metadata."""
        texts_tenant_A = ["foo_A", "bar_A", "baz_A"]
        texts_tenant_B = ["foo_B", "bar_B", "baz_B"]
        docsearch_TA = Weaviate.from_texts(
            texts_tenant_A,
            embedding_openai,
            weaviate_url=weaviate_url,
            tenant="tenantA",
            index_name="MultiTenantClass"
        )
        docsearch_TB = Weaviate.from_texts(
            texts_tenant_B,
            embedding_openai,
            weaviate_url=weaviate_url,
            tenant="tenantB",
            index_name="MultiTenantClass"
        )
        output_a = docsearch_TA.similarity_search("foo", k=1, tenant="tenantA")
        assert output_a == [Document(page_content="foo_A")]        

        output_b = docsearch_TB.similarity_search("foo", k=1,  tenant="tenantB")
        assert output_b == [Document(page_content="foo_B")]          