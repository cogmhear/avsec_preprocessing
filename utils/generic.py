
def check_flags(all_feat, face_embed, landmark, lip_embed, lip_images):
    assert (all_feat or landmark or lip_images or lip_embed or face_embed) == True, \
        "Please select atleast one feature to extract"
    if all_feat:
        landmark = True
        lip_images = True
        lip_embed = True
        face_embed = True
    elif lip_embed:
        assert lip_images == True, "Lip embed option requires lip_embed to be True"
        landmark = True
    return face_embed, landmark, lip_embed, lip_images