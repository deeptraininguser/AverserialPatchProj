import cv2
import torch

border_size = 2

displayed_aruco_code = 3

marker_size = 5000  # Size in pixels

aruco_dict_type = cv2.aruco.DICT_4X4_50  # ArUco dictionary type

latent_size = 4  # Spatial size of the latent (latent shape will be [batch, 4, latent_size, latent_size])

latent_batch_size = 10  # Number of patches in the latent batch


# orig_clases = torch.tensor([954]).cuda()  # banana
# orig_clases = torch.tensor([898,804]).cuda()  # purse, mailbag, backpack
# orig_clases = torch.tensor([899,849,505]).cuda() #water jug, teapot, coffepot
# orig_clases = torch.tensor([420, 402, 546, 889]).cuda() # guitar
# orig_clases = torch.tensor([504,968,899]).cuda()  # coffee mug
# orig_clases = torch.tensor([700,999]).cuda()# paper towel, toilet tissue
# orig_clases = torch.tensor([636,414,748]).cuda()# mail bag, backpack, purse
orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717, 866, 661, 864])  # jeep
# orig_clases = torch.tensor([852,722,574]).cuda()

hard_targets = { 1: 'goldfish, Carassius auratus',
7: 'cock',
21: 'kite',
207: 'golden retriever',
340: 'zebra',
745: 'projector',
779: 'school bus',
846: 'table lamp',
947: 'mushroom',
950: 'orange'}
