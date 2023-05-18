docker cp ./pips/highway.json 5bb65134181a:/cpp-pips/examples && \
docker exec 5bb65134181a /cpp-pips/run_ldips.sh && \
docker exec 5bb65134181a python3 /cpp-pips/solution/translate.py && \
docker cp 5bb65134181a:/cpp-pips/learned_policy.py ./pips
