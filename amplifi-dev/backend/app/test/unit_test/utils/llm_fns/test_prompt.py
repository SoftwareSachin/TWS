from app.utils.llm_fns.prompt import get_system_prompt, get_user_prompt


def test_get_user_prompt():
    assert (
        get_user_prompt("What's blue?", ["The sky is blue", "The sea is blue"])
        == """Answer the question based on the context below. 
Context:


---

The sky is blue

---

The sea is blue

Question: What's blue?
Answer:"""
    )


def test_get_system_prompt():
    assert (
        get_system_prompt()
        == "You are a helpful assistant that answers queries based on context."
    )
