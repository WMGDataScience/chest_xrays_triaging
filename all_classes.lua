local csv_dir_name = split_string(opt.input_csv_dir,'/')[#split_string(opt.input_csv_dir,'/')]

if csv_dir_name == "split_v6_26_Oct_16_mauro" then
classes = { "object"
    ,"normal"
    ,"pleural_effusion"
    ,"consolidation"
    ,"lung_nodule"
    ,"cardiomegaly"
    ,"atelectasis"
    ,"pneumothorax"
    ,"airspace_shadowing"
    ,"lung_mass"
    ,"hyperexpanded_lungs"
    ,"subcutaneous_emphysema"
    ,"pleural_thickening"
    ,"interstitial_shadowing"
    ,"pleural_lesion"
    ,"rib_fracture"
    ,"scoliosis"
    ,"unfolded_aorta"
    ,"hernia"
    ,"ground_glass_opacification"
    ,"widened_paratracheal_stripe"
    ,"left_lower_lobe_collapse"
    ,"right_upper_lobe_collapse"
    ,"bronchial_wall_thickening"
    ,"cavitating_lung_lesion"
    ,"bulla"
    ,"tension_pneumothorax"
    ,"pneumoperitoneum"
    ,"left_upper_lobe_collapse"
    ,"enlarged_hilum"
    ,"dilated_bowel"
    ,"right_lower_lobe_collapse"
    ,"rib_lesion"
    ,"mediastinum_widened"
    ,"aortic_calcification"
    ,"pneumomediastinum"
    ,"clavicle_fracture"
    ,"dextrocardia"
    ,"hemidiaphragm_elevated"
    ,"right_middle_lobe_collapse"
    ,"mediastinum_displaced"
}
elseif csv_dir_name == "split_v7_13_Dec_16_mauro"  then
classes = { 'consolidation',
 'object',
 'normal',
 'cardiomegaly',
 'pleural_thickening',
 'parenchymal_lesion',
 'pleural_effusion',
 'pneumothorax',
 'right_upper_lobe_collapse',
 'atelectasis',
 'cavitating_lung_lesion',
 'rib_fracture',
 'widened_paratracheal_stripe',
 'bronchial_wall_thickening',
 'scoliosis',
 'hernia',
 'pleural_lesion',
 'hyperexpanded_lungs',
 'interstitial_shadowing',
 'pneumomediastinum',
 'left_lower_lobe_collapse',
 'unfolded_aorta',
 'subcutaneous_emphysema',
 'mediastinum_widened',
 'clavicle_fracture',
 'enlarged_hilum',
 'dilated_bowel',
 'ground_glass_opacification',
 'left_upper_lobe_collapse',
 'right_lower_lobe_collapse',
 'dextrocardia',
 'pneumoperitoneum',
 'mediastinum_displaced',
 'bulla',
 'aortic_calcification',
 'right_middle_lobe_collapse',
 'rib_lesion',
 'hemidiaphragm_elevated'
}
elseif csv_dir_name == "split_v8_6_Jan_17_mauro" or csv_dir_name == "split_v9_16_Jan_17_mauro" then
classes = { 'consolidation',
 'object',
 'normal',
 'cardiomegaly',
 'emphysema',
 'pleural_thickening',
 'parenchymal_lesion',
 'pleural_effusion',
 'pneumothorax',
 'right_upper_lobe_collapse',
 'atelectasis',
 'cavitating_lung_lesion',
 'rib_fracture',
 'widened_paratracheal_stripe',
 'bronchial_wall_thickening',
 'scoliosis',
 'hernia',
 'pleural_lesion',
 'hyperexpanded_lungs',
 'interstitial_shadowing',
 'pneumomediastinum',
 'left_lower_lobe_collapse',
 'unfolded_aorta',
 'subcutaneous_emphysema',
 'mediastinum_widened',
 'clavicle_fracture',
 'enlarged_hilum',
 'dilated_bowel',
 'ground_glass_opacification',
 'left_upper_lobe_collapse',
 'right_lower_lobe_collapse',
 'dextrocardia',
 'pneumoperitoneum',
 'mediastinum_displaced',
 'bulla',
 'aortic_calcification',
 'right_middle_lobe_collapse',
 'rib_lesion',
 'hemidiaphragm_elevated'
}
elseif csv_dir_name == "split_v10_15_Mar_17_mauro" or csv_dir_name == "split_v12_27_Jun_17_mauro" then
classes = { 
 'consolidation',
 'object',
 'normal',
 'cardiomegaly',
 'emphysema',
 'pleural_abnormality',
 'parenchymal_lesion',
 'pleural_effusion',
 'pneumothorax',
 'right_upper_lobe_collapse',
 'atelectasis',
 'cavitating_lung_lesion',
 'rib_fracture',
 'bronchial_wall_thickening',
 'scoliosis',
 'hernia',
 'hyperexpanded_lungs',
 'interstitial_shadowing',
 'pneumomediastinum',
 'left_lower_lobe_collapse',
 'unfolded_aorta',
 'subcutaneous_emphysema',
 'mediastinum_widened',
 'clavicle_fracture',
 'paratracheal_hilar_enlargement',
 'dilated_bowel',
 'ground_glass_opacification',
 'left_upper_lobe_collapse',
 'right_lower_lobe_collapse',
 'dextrocardia',
 'pneumoperitoneum',
 'mediastinum_displaced',
 'bulla',
 'aortic_calcification',
 'right_middle_lobe_collapse',
 'rib_lesion',
 'hemidiaphragm_elevated'
}

elseif csv_dir_name == "split_v11_7_June_7_emanuele" or csv_dir_name == "split_v13_4_July_emanuele" or csv_dir_name == "v11"  then
classes = { 
 'abnormal_other',
 'airspace_opacifation',
 'bone_abnormality',
 'cardiomegaly',
 'collapse',
 'hiatus_hernia',
 'interstitial_shadowing',
 'intraabdominal_pathology',
 'object',
 'normal',
 'paratracheal_hilar_enlargement',
 'parenchymal_lesion',
 'pleural_abnormality',
 'pneumomediastinum',
 'pneumothorax',
 'subcutaneous_emphysema'
}

elseif csv_dir_name == "original_labels" then
classes = { 
 'Atelectasis',
 'Cardiomegaly',
 'Effusion',
 'Infiltration',
 'Mass',
 'Nodule',
 'Pneumonia',
 'Pneumothorax',
 'Consolidation',
 'Edema',
 'Emphysema',
 'Fibrosis',
 'Pleural_Thickening',
 'Hernia',
 'normal'
}

end
    
if opt.delete_object_class then
    table.find_and_remove(classes,"object")
end
