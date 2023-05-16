docker cp ./pips/highway.json c81f9ae9ac8c:/cpp-pips/examples && \
docker exec c81f9ae9ac8c /cpp-pips/run_ldips.sh && \
docker exec c81f9ae9ac8c python3 /cpp-pips/solution/translate.py && \
docker cp c81f9ae9ac8c:/cpp-pips/learned_policy.py ./pips/
