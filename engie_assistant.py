import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


class EngieAssistant:
    def __init__(
        self, customer_data, system_prompt, llm, history=[], vector_store=None
    ):
        self.customer = customer_data
        self.system_prompt = system_prompt
        self.llm = llm
        self.messages = history
        self.vector_store = vector_store
        self.employee_information = customer_data

        self.chain = self.get_conversation_chain()

    def get_conversation_chain(self):
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("conversation_history"),
                ("human", "{user_input}"),
            ]
        )
        llm = self.llm
        output_parser = StrOutputParser()

        chain = (
            {
                "retrieved_policy_information": self.vector_store.as_retriever(),
                "employee_information": lambda x: self.employee_information,
                "user_input": RunnablePassthrough(),
                "conversation_history": lambda x: self.messages,
            }
            | prompt
            | llm
            | output_parser
        )
        return chain

    def get_response(self, user_input):
        return self.chain.stream(user_input)

    def render_messages(self):
        messages = self.messages

        for message in messages:
            if message["role"] == "user":
                st.chat_message("human").markdown(message["content"])
            if message["role"] == "ai":
                st.chat_message("ai").markdown(message["content"])

    def set_state(self, key, value):
        st.session_state[key] = value

    def render_user_input(self):

        user_input = st.chat_input("Type here...", key="input")
        if user_input and user_input != "":
            st.chat_message("human").markdown(user_input)
            self.messages.append({"role": "user", "content": user_input})

            response_generator = self.get_response(user_input)

            with st.chat_message("ai"):
                response = st.write_stream(response_generator)

            self.messages.append({"role": "ai", "content": response})

            self.set_state("messages", self.messages)

    def render(self):

        with st.sidebar:
            st.logo(
                "https://upload.wikimedia.org/wikipedia/commons/0/0e/Umbrella_Corporation_logo.svg"
            )
            st.title("Umbrella Corporation Assistant")

            st.subheader("Employee Information")
            st.write(self.employee_information)

        self.render_messages()
        self.render_user_input()
