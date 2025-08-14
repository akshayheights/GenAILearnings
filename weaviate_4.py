from modules import *
class WeaviateFieldClusterer:
    def __init__(self, cluster_url_weaviate, api_key_weaviate, model_name, input_file_name, embedding_dict, thresholds,
                 base_url_llm, api_key_llm,output_file_name):
        self.file_name = input_file_name
        self.df = pd.read_csv(input_file_name)
        self.model = SentenceTransformer(model_name)
        self.embedding_dict = embedding_dict
        self.thresholds = thresholds
        self.output_file_name = output_file_name

        # Weaviate client
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url_weaviate,
            auth_credentials=api_key_weaviate
        )
        self.client.connect()

        # OpenRouter LLM client (via OpenAI interface)
        self.llm_client = OpenAI(
            base_url=base_url_llm,
            api_key=api_key_llm
        )

        self.grouped_records = set()

    def create_collection(self, class_name):
        if not self.client.collections.exists(class_name):
            self.client.collections.create(
                name=class_name,
                properties=[
                    Property(name="record_id", data_type=DataType.TEXT),
                    Property(name="field_value", data_type=DataType.TEXT)
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )

    def create_store_embeddings(self):
        for collection_name in self.embedding_dict:
            self.create_collection(collection_name)

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Storing embeddings"):
            record_id = row["record_id"]
            for collection_name, field_name in self.embedding_dict.items():
                field_value = str(row[field_name])
                vector = self.model.encode(field_value)
                self.client.collections.get(collection_name).data.insert(
                    properties={"record_id": record_id, "field_value": field_value},
                    vector=vector

                )

    def close_connection(self):
        self.client.close()

    def get_similar_records(self, collection_name, vector, threshold, leader_id):
        results = self.client.collections.get(collection_name).query.near_vector(
            near_vector=vector,
            limit=10,
            return_properties=["record_id", "field_value"],
            certainty=1 - threshold
        )
        return [
            {
                "record_id": o.properties["record_id"],
                "field_value": o.properties["field_value"]
            }
            for o in results.objects
            if o.properties["record_id"] != leader_id and o.properties["record_id"] not in self.grouped_records
        ]

    def is_similar_llm(self, leader_fields: dict, candidate_fields: dict) -> tuple:
        # print(f"leader_fields:{leader_fields}")
        # print(f"candidate_fields:{candidate_fields}")
        prompt = f"""
                    You are a data deduplication assistant. You will be shown two records with multiple fields such as:

                    - fname (first name)
                    - lname (last name)
                    - stNo (street number)
                    - add1 (address line 1)
                    - add2 (address line 2)
                    - city
                    - areacode
                    - state
                    - dob (date of birth)
                    - ssn (social security number)

                    Your job is to determine if the two records refer to the **same real-world person**. Consider:
                    - Typos, misspellings, or slight variations in names or addresses
                    - Abbreviations and common format differences (e.g., "St" vs "Street", "VIC" vs "Victoria")
                    - Date format differences or digit typos
                    - Semantic or contextual similarity between addresses
                    - Uniqueness of SSN and DOB if available

                    Leader Record:
                    {leader_fields}

                    Candidate Record:
                    {candidate_fields}

                    Respond ONLY with:
                    - "YES - <reason in one short sentence>" if they refer to the same person
                    - "NO - <reason in one short sentence>" if they do not
                    """
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content.strip()

        if response.lower().startswith("yes"):
            return True, response
        return False, response

    def group_similar_records_llm(self):
        record_to_group = {}
        record_to_reason = {}
        group_id_counter = 1

        for _, leader_row in tqdm(self.df.iterrows(), total=len(self.df), desc="Grouping Leaders"):
            leader_id = leader_row["record_id"]
            if leader_id in record_to_group:
                continue

            leader_fields = {
                field_name: leader_row[field_name]
                for field_name in self.embedding_dict.values()
            }

            group_members = {leader_id}
            group_reasons = {}
            seen_candidates = set()

            for collection_name, field_name in tqdm(self.embedding_dict.items(), desc="Field Loop", leave=False):
                start_enc = time.perf_counter()
                vector = self.model.encode(str(leader_row[field_name]))
                print(f"Encoding [{field_name}] took {time.perf_counter() - start_enc:.4f} sec")

                start_sim = time.perf_counter()
                candidates = self.get_similar_records(
                    collection_name, vector, self.thresholds[collection_name], leader_id
                )
                print(f"Similarity Search on [{collection_name}] took {time.perf_counter() - start_sim:.4f} sec")

                candidate_tasks = []
                with ThreadPoolExecutor(max_workers=30) as executor:
                    for candidate in candidates:
                        cid = candidate["record_id"]
                        if cid in record_to_group or cid in seen_candidates:
                            continue
                        seen_candidates.add(cid)

                        def llm_task(cid=cid):
                            candidate_fields = {
                                field_name: self.df.loc[self.df["record_id"] == cid, field_name].values[0]
                                for field_name in self.embedding_dict.values()
                            }
                            return cid, *self.is_similar_llm(leader_fields, candidate_fields)

                        candidate_tasks.append(executor.submit(llm_task))

                    for future in as_completed(candidate_tasks):
                        cid, similar, reason = future.result()
                        if similar:
                            group_members.add(cid)
                            group_reasons[cid] = reason

            group_id = f"group-{group_id_counter:04d}"
            for rid in group_members:
                record_to_group[rid] = group_id
                record_to_reason[rid] = group_reasons.get(rid, "Leader of the group")
                self.grouped_records.add(rid)
            group_id_counter += 1

        self.df["group_id"] = self.df["record_id"].map(record_to_group)
        self.df["grouping_reason"] = self.df["record_id"].map(record_to_reason)
        self.df = self.df.sort_values(by="group_id").reset_index(drop=True)
        self.df.to_csv(self.output_file_name)

        return self.df[["record_id", *self.embedding_dict.values(), "group_id", "grouping_reason"]]

