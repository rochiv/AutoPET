seg.shape
seg.mask()
seg.min()
outputs.unique()
out = torch.where(outputs > 0.5, 1., 0.)
dice_score(seg, out)
net = load('model.pt')
new_model = model.load_state_dict(net)
new_model
new_out = model(pet_suv.float())
new_out.max()
Out[22]: tensor(1.3895, device='cuda:0')
new_out.min()
Out[23]: tensor(-12.0866, device='cuda:0')
new_out = torch.where(new_out>0.5, 1., 0.)
new_out.max()
Out[25]: tensor(1., device='cuda:0')
new_out.min()
Out[26]: tensor(0., device='cuda:0')
(seg*new_out).sum()
Out[27]: tensor(27., device='cuda:0')
dice_score(seg,new_out)
Out[28]: tensor(0.3214, device='cuda:0')
dice_score(seg[0],new_out[0])
Out[29]: tensor(0.3624, device='cuda:0')
dice_score(seg[1],new_out[1])
Out[30]: tensor(0., device='cuda:0')
seg.shape
Out[31]: torch.Size([2, 1, 64, 64, 64])
dice_score(seg[1][0],new_out[1][0])
Out[32]: tensor(0., device='cuda:0')
seg[1].max()
Out[33]: tensor(1., device='cuda:0')
new_out[1].max()
Out[34]: tensor(0., device='cuda:0')
new_out[1].min()
Out[35]: tensor(0., device='cuda:0')
seg.shape
Out[36]: torch.Size([2, 1, 64, 64, 64])
sample[0].shape
Out[39]: torch.Size([2, 1, 64, 64, 64])
pet_suv.shape
Out[40]: torch.Size([2, 2, 64, 64, 64])
new_out = model(pet_suv.float())
import napari
napari.viewer()
viewer = napari.Viewer()
viewer.add_image(new_out[0][0].cpu().numpy())
Out[45]: <Image layer 'Image' at 0x7f2e1beef880>
viewer.add_labels(seg[0][0].cup().numpy())
viewer.add_labels(seg[0][0].cpu().numpy().astype('uint8'))
Out[48]: <Labels layer 'Labels' at 0x7f2e1137e080>
new_out = torch.where(new_out>0.5, 1., 0.)
viewer.add_image(new_out[0][0].cpu().numpy())
Out[50]: <Image layer 'Image [1]' at 0x7f2e1076ffa0>
viewer.add_labels(seg[0][0].cpu().numpy().astype('uint8'))
Out[51]: <Labels layer 'Labels [1]' at 0x7f2e107cf640>
viewer.add_image(new_out[0][0].cpu().numpy())
Out[52]: <Image layer 'Image' at 0x7f2e1bd92d10>
viewer.add_labels(seg[0][0].cpu().numpy().astype('uint8'))
Out[53]: <Labels layer 'Labels' at 0x7f2e10c661a0>
viewer.add_image(new_out[1][0].cpu().numpy())
Out[54]: <Image layer 'Image' at 0x7f2e10d1a0e0>
new_out = model(pet_suv.float())
viewer.add_image(new_out[1][0].cpu().numpy())
Out[56]: <Image layer 'Image' at 0x7f2dd85e3ca0>
new_out[1][0].max()
Out[57]: tensor(0.2242, device='cuda:0')
new_out = torch.where(new_out>0.1, 1., 0.)
viewer.add_image(new_out[1][0].cpu().numpy())
Out[59]: <Image layer 'Image [1]' at 0x7f2dd8f8f040>
viewer.add_image(new_out[1][0].cpu().numpy())
Out[60]: <Image layer 'Image' at 0x7f2dd8690550>
Out[61]: <Image layer 'Image [1]' at 0x7f2e10fa5780>
viewer.add_labels(seg[1][0].cpu().numpy().astype('uint8'))
Out[62]: <Labels layer 'Labels' at 0x7f2e107cc850>
viewer.add_image(seg[1][0].cpu().numpy().astype('uint8'))
Out[63]: <Image layer 'Image [2]' at 0x7f2e1becf4f0>
viewer = napari.Viewer()
viewer.add_image(seg[1][0].cpu().numpy().astype('uint8'))
Out[65]: <Image layer 'Image' at 0x7f2e10d1ae90>
viewer = napari.Viewer()
viewer.add_image(new_out[1][0].cpu().numpy())
Out[67]: <Image layer 'Image' at 0x7f2e101556c0>
viewer.add_labels(seg[1][0].cpu().numpy().astype('uint8'))
Out[68]: <Labels layer 'Labels' at 0x7f2da2385a20>

