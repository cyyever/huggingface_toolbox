from typing import Any

import torch
import transformers
from cyy_torch_toolbox import TokenIDsType, TokenIDType, TokenizerMixin


class HuggingFaceTokenizer(TokenizerMixin):
    def __init__(self, tokenizer_config: dict) -> None:
        self.tokenizer_config = tokenizer_config.copy()
        # # # self.tokenizer.padding_side = "left"

    @property
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

    def get_tokens_from_transformed_result(self, transformed_result: Any) -> list[str]:
        batch_encoding = self.__get_batch_encoding_from_transformed_result(
            transformed_result
        )
        return batch_encoding.tokens()

    def get_token_id(self, token: str) -> int | list[int]:
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_token(self, token_id: TokenIDType) -> str:
        return self.tokenizer.decode(token_id)
