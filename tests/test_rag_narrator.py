from rag_engine.llm.rag_narrator import RAGNarrativeGenerator


class CapturingLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens=512):
        self.prompts.append(prompt)
        return 'ok'


class FakeSearch:
    def __init__(self):
        self.queries = []

    def find_related(self, query, k=10):
        self.queries.append(query)
        return []


def test_without_a_semantic_index_the_prompt_names_the_static_analysis_facts_as_its_context():
    llm = CapturingLLM()
    narrator = RAGNarrativeGenerator(llm, None)
    narrator.answer_question('What is X?', 'FACT-LINE')
    narrator.generate('query', 'SUMMARY-LINE')
    for prompt in llm.prompts:
        assert 'FAISS' not in prompt and 'RETRIEVED CODE CONTEXT' not in prompt
        assert 'No related functions' not in prompt and 'semantic search index' not in prompt
    question_prompt = llm.prompts[0]
    assert '=== STATIC-ANALYSIS FACTS ===\nFACT-LINE' in question_prompt
    assert 'Answer based on the static-analysis facts' in question_prompt


def test_with_a_semantic_index_the_prompt_still_carries_the_retrieved_code():
    llm, search = CapturingLLM(), FakeSearch()
    RAGNarrativeGenerator(llm, search).answer_question('What is X?', 'FACT-LINE')
    assert search.queries == ['What is X?']
    assert 'RETRIEVED CODE CONTEXT (FAISS semantic search)' in llm.prompts[0]
    assert '=== ANALYSIS SUMMARY ===\nFACT-LINE' in llm.prompts[0]
