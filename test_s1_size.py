import ee

ee.Initialize(project='astute-baton-486304-r1')
roi = ee.Geometry.Rectangle([80.20, 12.92, 80.25, 12.98])
col = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW')))

print("Pre-event (Oct 1 - Nov 20):", col.filterDate('2023-10-01', '2023-11-20').size().getInfo())
print("Post-event (Dec 5 - Dec 20):", col.filterDate('2023-12-05', '2023-12-20').size().getInfo())
print("Expanded Post (Dec 5 - Dec 31):", col.filterDate('2023-12-05', '2023-12-31').size().getInfo())
