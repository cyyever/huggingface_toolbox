from collections.abc import Mapping
from cyy_naive_lib.log import log_debug
from functools import cached_property
from typing import Any

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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        assert self.tokenizer.is_fast

    def __call__(
        self, *args, nested_batch_encoding: bool = False, **kwargs
    ) -> transformers.BatchEncoding | list[transformers.BatchEncoding]:
        res = self.__tokenizer(*args, **kwargs)
        if nested_batch_encoding:
            return [res]
        return res

    @cached_property
    def special_tokens(self) -> set[str]:
        tokens = set()
        for attr in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            if attr != "additional_special_tokens" and hasattr(self.tokenizer, attr):
                special_token = getattr(self.tokenizer, attr)
                if special_token is not None:
                    tokens.add(special_token)
        return tokens

    @cached_property
    def special_token_ids(self) -> set:
        ids: set = set()
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
        return self.tokenizer.get_vocab()

    def get_mask_token(self) -> str | list[str] | None:
        return self.tokenizer.mask_token

    def tokenize(self, phrase: str) -> transformers.BatchEncoding:
        return self.tokenizer(phrase, return_tensors="pt", truncation=False)

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

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        batch_encoding = self.__get_batch_encoding_from_transformed_result(
            transformed_result
        )
        return batch_encoding.tokens()

    def get_token_id(self, token: str) -> int | list[int]:
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_token(self, token_id: TokenIDType) -> str:
        return self.tokenizer.decode(token_id)

    def strip_special_tokens(self, token_ids: TokenIDsType) -> TokenIDsType:
        for special_token_id in self.special_token_ids:
            token_ids = token_ids[token_ids != special_token_id]
        return token_ids
