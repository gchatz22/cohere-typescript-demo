import { CohereClient } from "cohere-ai";
import { EmbedByTypeResponseEmbeddings } from "cohere-ai/api";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { LocalIndex } from "vectra";
import wiki from "wikipedia";

/*
Based off https://github.com/cohere-ai/notebooks/blob/main/notebooks/Vanilla_RAG.ipynb
Execute
1. yarn
2. ts-node cohere-rag.ts
*/

const main = async () => {
  const co = new CohereClient({
    token: "",
  });

  // fetch wikipedia page
  const page = await wiki.page("Dune Part Two");
  const text = await page.content();
  console.log(`The text has roughly ${text.split(" ").length} words.`);

  // chunk document
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 50,
    lengthFunction: (s: string) => s.length,
  });
  const chunks_ = await splitter.createDocuments([text]);
  const chunks = chunks_.map((chunk) => chunk.pageContent);
  console.log(`The text has been broken down in ${chunks.length} chunks.`);

  // embed page
  const model = "embed-english-v3.0";
  const embedChunks = createChunks(chunks);
  const embeddingsTexts = (
    await Promise.all(
      embedChunks.map((embedChunk) =>
        co.embed({
          texts: embedChunk,
          model: model,
          inputType: "search_document",
          embeddingTypes: ["float"],
        })
      )
    )
  )
    .flat()
    .reduce(
      (acc, response) => {
        return {
          texts: [...acc.texts, ...response.texts],
          embeddings: [
            ...acc.embeddings,
            ...(response.embeddings as EmbedByTypeResponseEmbeddings).float!,
          ],
        };
      },
      { texts: [], embeddings: [] } as {
        texts: string[];
        embeddings: number[][];
      }
    );
  console.log(`Processed ${embeddingsTexts.embeddings.length} embeddings`);

  // create index and insert embeddings
  const index = new LocalIndex("./index");
  if (await index.isIndexCreated()) {
    await index.deleteIndex();
  }
  await index.createIndex();
  await Promise.all(
    embeddingsTexts.texts.map((text, i) =>
      index.insertItem({
        vector: embeddingsTexts.embeddings[i],
        metadata: { text },
      })
    )
  );

  // embed query
  const query =
    "Name everyone involved in writing the script, directing, and producing 'Dune: Part Two'?";
  const queryEmbedding = (
    (
      await co.embed({
        texts: [query],
        model: model,
        inputType: "search_query",
        embeddingTypes: ["float"],
      })
    ).embeddings as EmbedByTypeResponseEmbeddings
  )["float"]![0];
  const queryDocuments = await index.queryItems(queryEmbedding, 10);
  const documents = queryDocuments.map((d) => {
    return { text: d.item.metadata["text"].toString() };
  });

  const chatResponse = await co.chat({
    model: "command-r-plus",
    message: query,
    documents: documents,
  });
  console.log("------Response");
  console.log(chatResponse.text);
  console.log("\n------Citations\n");
  console.log(chatResponse.citations);
  console.log("\n------Documents\n");
  console.log(chatResponse.documents);
};

const createChunks = (arr: string[], size: number = 96) => {
  const chunkedArray: string[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunkedArray.push(arr.slice(i, i + size));
  }
  return chunkedArray;
};

main().catch((e) => console.log(e));
