from collections.abc import Mapping
from functools import cached_property
from typing import Any
from transformers import PreTrainedTokenizerFast


import torch
import transformers
from cyy_torch_toolbox import TokenIDsType, TokenIDType, Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, tokenizer_config: dict) -> None:
        self.__tokenizer: transformers.PreTrainedTokenizerFast = (
            transformers.AutoTokenizer.from_pretrained(
                tokenizer_config["name"],
                **tokenizer_config.get("kwargs", {}),
                use_fast=True,
            )
        )
        if not self.__tokenizer.is_fast:
            self.__tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self.__tokenizer
            )

    @cached_property
    def special_tokens(self) -> set[str]:
        tokens = set()
        for attr in self.__tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            if attr != "additional_special_tokens" and hasattr(self.__tokenizer, attr):
                special_token = getattr(self.__tokenizer, attr)
                if special_token is not None:
                    tokens.add(special_token)
        return tokens

    @cached_property
    def special_token_ids(self) -> set[int | tuple[int]]:
        ids: set[int | tuple[int]] = set()
        for token in self.special_tokens:
            res: int | list[int] = self.get_token_id(token)
            if isinstance(res, list):
                ids.add(tuple(res))
            else:
                ids.add(res)
        return ids

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.__tokenizer

    def get_vocab(self) -> Mapping[str, int]:
        return self.__tokenizer.get_vocab()

    def get_mask_token(self) -> str:
        return self.__tokenizer.mask_token

    def tokenize(self, phrase: str) -> list[str]:
        encoding = self.__tokenizer(phrase, return_tensors="pt", truncation=False)
        return [
            token for token in encoding.tokens() if token not in self.special_tokens
        ]

    def get_token_ids_from_transformed_result(
        self, transformed_result: Any
    ) -> TokenIDsType:
        assert isinstance(transformed_result, transformers.BatchEncoding)
        input_ids_tensor = transformed_result["input_ids"]
        assert isinstance(input_ids_tensor, torch.Tensor)
        return input_ids_tensor.squeeze()

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        assert isinstance(transformed_result, transformers.BatchEncoding)
        return transformed_result.tokens()

    def get_token_id(self, token: str) -> int | list[int]:
        return self.__tokenizer.convert_tokens_to_ids(token)

    def get_token(self, token_id: TokenIDType) -> str:
        return self.__tokenizer.decode(token_id)

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        for special_token_id in self.special_token_ids:
            token_ids = token_ids[token_ids != special_token_id]
        return token_ids
