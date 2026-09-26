# Hypothesis
The models does some reasoning better based on images. Therefore we should either supply the images directly to the model, or do it in two stages to extract new insights through an image-only prompt that we then supply to a second stage text-only prompt that solves the problem.

# Images

## 64efde09 example images

<img width="1044" height="468" alt="64efde09_first_pair_cartoon" src="https://github.com/user-attachments/assets/55b8644f-f622-4b88-9c55-d87d51b14899" />

<img width="1044" height="846" alt="64efde09_cartoon" src="https://github.com/user-attachments/assets/906c8f02-1339-48e7-8ecd-22e562eb2519" />

<img width="1044" height="1368" alt="64efde09_with_test_cartoon" src="https://github.com/user-attachments/assets/fe010c8f-734d-4fbf-a6f5-b5fbd743f8ab" />

<img width="1044" height="468" alt="64efde09_first_pair_precise" src="https://github.com/user-attachments/assets/0a68a555-4cdd-492a-85f2-ab04b423c23e" />

<img width="1044" height="846" alt="64efde09_precise" src="https://github.com/user-attachments/assets/6dff137d-5036-41b4-9bfd-dc9d3e6df3a1" />

<img width="1044" height="1368" alt="64efde09_with_test_precise" src="https://github.com/user-attachments/assets/7b9480e2-673b-45f0-bb21-296ce6b6c7fb" />

## 8f3a5a89 example images

<img width="576" height="288" alt="8f3a5a89_first_pair_cartoon" src="https://github.com/user-attachments/assets/1b15ddfc-25e6-42e1-a660-7a5518573349" />

<img width="720" height="827" alt="8f3a5a89_cartoon" src="https://github.com/user-attachments/assets/a1fa4aac-9822-4bcf-9ee6-9480260fbe84" />

<img width="720" height="1044" alt="8f3a5a89_with_test_cartoon" src="https://github.com/user-attachments/assets/bf50c4b8-3879-49ed-85b8-796ab2346021" />

<img width="576" height="288" alt="8f3a5a89_first_pair_precise" src="https://github.com/user-attachments/assets/545bf372-07c8-446c-b7c6-a407d8f47fa8" />

<img width="720" height="827" alt="8f3a5a89_precise" src="https://github.com/user-attachments/assets/980f92fb-5b11-4a10-ac3a-49233b13203f" />

<img width="720" height="1044" alt="8f3a5a89_with_test_precise" src="https://github.com/user-attachments/assets/f8d37002-5312-46b7-82b4-8b578785365d" />

## 332f06d7 example images


<img width="432" height="216" alt="332f06d7_first_pair_cartoon" src="https://github.com/user-attachments/assets/594d5ba2-39b2-48e3-a53a-2566b37cc2a8" />
<img width="432" height="216" alt="332f06d7_first_pair_precise" src="https://github.com/user-attachments/assets/f90f309f-8d6f-42ed-9924-e0aca971b1c8" />
<img width="720" height="1296" alt="332f06d7_with_test_cartoon" src="https://github.com/user-attachments/assets/f077e653-ea75-4b7d-afa5-4adadeff82ad" />
<img width="720" height="1296" alt="332f06d7_with_test_precise" src="https://github.com/user-attachments/assets/a171167e-9d30-49cd-bbc6-43f46ee6b87d" />




# Performance - single multimodal prompt

I have seen multimodality help in some cases, but it's not consistent and in somewhat structured testingi I can't really recreate it

## 64efde09

This problem gets solved by my text-only solver every 1 out of 4 runs or so. It's a hard problem.

- 64efde09_cartoon.png: Failed
- 64efde09_with_test_precise.png: Failed
- 64efde09_precise.png: Failed
- 64efde09_first_pair_precise.png: Failed

## 332f06d7

This is a problem that my solver pretty much can't solve, or possibly solves very rarely.

- 332f06d7_cartoon.png: Failed
- 332f06d7_precise.png: Failed
- 332f06d7_first_pair_precise.png: Failed
- 332f06d7_with_test_precise.png: Failed


# Conclusion

We probably instead should extract the hints in a separate prompt and stay text-only


