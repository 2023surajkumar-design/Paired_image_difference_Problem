# Image difference report `20261012-164750-2924ab`

**Comparison status:** comparable_with_warnings  
**Alignment:** similarity (reliable)
**Pair warnings:** focus_or_blur_differs_between_images  
**Components that ran:** dinov2, matcher, lowlevel, ocr, vlm, matcher:lightglue, vlm_hypotheses  

### Accepted changes (2)
- `claim_012` localized visual change (no semantic explanation) at A[468, 347, 472, 356]. Evidence: appearance. very strong persistent pixel change under usable alignment, no nuisance signature; reported as a localized change WITHOUT semantic explanation
- `claim_056` added: dot (? -> orange dot on thorax) at A[177, 41, 205, 61]. Evidence: appearance, vlm. strong persistent pixel change under usable alignment + independent support from vlm

### Needs review (45)
- `claim_001` added: red dot (green striped background -> green striped background with a red dot) at A[121, 71, 168, 125]. Evidence: appearance, vlm. nuisance explanation plausible: local_displacement_explains_residual(misregistration or moved content); independent support from ['vlm']
- `claim_002` added: orange circle (black and white butterfly wing with no added circle -> black and white butterfly wing with an orange circle superimposed on it) at A[215, 79, 336, 188]. Evidence: appearance, vlm. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow; independent support from ['vlm']
- `claim_006` localized visual change (no semantic explanation) at A[294, 105, 450, 211]. Evidence: appearance. strong persistent pixel change but VLM crop re-inspection reports no content change
- `claim_009` localized visual change (no semantic explanation) at A[170, 28, 249, 102]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_010` localized visual change (no semantic explanation) at A[170, 25, 294, 47]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_011` localized visual change (no semantic explanation) at A[163, 194, 234, 262]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_013` localized visual change (no semantic explanation) at A[124, 114, 187, 149]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_014` localized visual change (no semantic explanation) at A[240, 186, 309, 261]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_015` localized visual change (no semantic explanation) at A[120, 115, 168, 148]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_016` localized visual change (no semantic explanation) at A[131, 121, 190, 170]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_019` localized visual change (no semantic explanation) at A[265, 25, 294, 46]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_020` localized visual change (no semantic explanation) at A[180, 261, 232, 290]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_021` localized visual change (no semantic explanation) at A[120, 129, 178, 171]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_022` localized visual change (no semantic explanation) at A[113, 262, 172, 290]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_023` localized visual change (no semantic explanation) at A[192, 152, 261, 254]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_024` localized visual change (no semantic explanation) at A[175, 173, 225, 228]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_025` localized visual change (no semantic explanation) at A[199, 275, 232, 290]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_026` localized visual change (no semantic explanation) at A[209, 41, 272, 96]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_027` localized visual change (no semantic explanation) at A[53, 138, 93, 165]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_028` localized visual change (no semantic explanation) at A[205, 58, 249, 112]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_029` localized visual change (no semantic explanation) at A[194, 200, 249, 254]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_030` localized visual change (no semantic explanation) at A[193, 55, 249, 115]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_031` localized visual change (no semantic explanation) at A[205, 94, 270, 162]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_032` localized visual change (no semantic explanation) at A[120, 117, 156, 157]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_034` localized visual change (no semantic explanation) at A[175, 179, 229, 234]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_035` localized visual change (no semantic explanation) at A[215, 180, 267, 245]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_036` localized visual change (no semantic explanation) at A[6, 187, 21, 234]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_037` localized visual change (no semantic explanation) at A[53, 138, 93, 165]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_038` localized visual change (no semantic explanation) at A[122, 115, 175, 166]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_039` localized visual change (no semantic explanation) at A[170, 28, 212, 66]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_040` localized visual change (no semantic explanation) at A[239, 213, 294, 261]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_041` localized visual change (no semantic explanation) at A[234, 138, 287, 194]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_042` localized visual change (no semantic explanation) at A[286, 108, 340, 163]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_043` localized visual change (no semantic explanation) at A[241, 25, 294, 66]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_044` localized visual change (no semantic explanation) at A[145, 261, 200, 288]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_045` localized visual change (no semantic explanation) at A[127, 114, 186, 166]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_046` localized visual change (no semantic explanation) at A[284, 119, 344, 163]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_047` localized visual change (no semantic explanation) at A[230, 141, 283, 196]. Evidence: appearance. nuisance explanation plausible: illumination_or_shadow
- `claim_048` localized visual change (no semantic explanation) at A[233, 32, 286, 75]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_049` localized visual change (no semantic explanation) at A[134, 113, 191, 155]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_050` localized visual change (no semantic explanation) at A[340, 193, 388, 221]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_051` localized visual change (no semantic explanation) at A[13, 153, 57, 198]. Evidence: appearance. pixel-only evidence of moderate strength
- `claim_052` localized visual change (no semantic explanation) at A[221, 198, 275, 248]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference, illumination_or_shadow
- `claim_053` localized visual change (no semantic explanation) at A[120, 116, 167, 157]. Evidence: appearance. nuisance explanation plausible: focus_blur_difference
- `claim_054` localized visual change (no semantic explanation) at A[399, 204, 403, 211]. Evidence: appearance. nuisance explanation plausible: exposure_clipping, focus_blur_difference, illumination_or_shadow, specular_or_saturation

### Unverified semantic observations (withheld)
- `claim_055` added: dot

### Plausible nuisance factors (not established facts)
- exposure_clipping
- focus_blur_difference
- illumination_or_shadow
- local_displacement_explains_residual(misregistration or moved content)
- specular_or_saturation

_9 candidate(s) abstained. Scores are uncalibrated unless a calibrator was applied._
