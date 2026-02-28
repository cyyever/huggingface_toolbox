from abc import abstractmethod
from typing import Any, override

import torch
import transformers
from cyy_torch_toolbox import TokenIDsType, TokenIDType, TokenizerMixin


class HuggingFaceTokenizerBase(TokenizerMixin):
    @property
    @abstractmethod
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast: ...

    @override
    def get_token_id(self, token: str) -> int | list[int]:
        return self.tokenizer.convert_tokens_to_ids(token)

    @override
    def get_token(self, token_id: TokenIDType) -> str:
        return self.tokenizer.convert_ids_to_tokens(token_id)


class HuggingFaceTokenizer(HuggingFaceTokenizerBase):
    def __init__(self, tokenizer_config: dict[str, Any]) -> None:
        self.tokenizer_config: dict[str, Any] = tokenizer_config.copy()
        self.__tokenizer: transformers.PreTrainedTokenizerFast | None = None
        # # # self.tokenizer.padding_side = "left"

    @override
    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_HuggingFaceTokenizer__tokenizer"] = None
        return state

    @property
    @override
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        if self.__tokenizer is None:
            self.__tokenizer: transformers.PreTrainedTokenizerFast = (
                transformers.AutoTokenizer.from_pretrained(
                    self.tokenizer_config["name"],
                    use_fast=True,
                    **self.tokenizer_config.get("kwargs", {}),
                )
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.__tokenizer

    def __get_batch_encoding_from_transformed_result(
        self, transformed_result: Any
    ) -> transformers.BatchEncoding:
        if isinstance(transformed_result, transformers.BatchEncoding):
            return transformed_result
        return self.tokenizer(transformed_result)

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
        batch_encoding = self.__get_batch_encoding_from_transformed_result(
            transformed_result
        )
        input_ids_tensor = batch_encoding["input_ids"]
        if isinstance(input_ids_tensor, list):
            input_ids_tensor = torch.tensor(input_ids_tensor)
        assert isinstance(input_ids_tensor, torch.Tensor)
        return input_ids_tensor.squeeze()

    @override
    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        batch_encoding = self.__get_batch_encoding_from_transformed_result(
            transformed_result
        )
        return batch_encoding.tokens()
