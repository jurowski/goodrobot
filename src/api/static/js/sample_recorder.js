function renderSample(sample) {
    const row = document.createElement('div');
    row.className = 'sample-row';
    
    // Create type badge
    const typeBadge = document.createElement('span');
    typeBadge.className = `sample-type ${sample.type.toLowerCase()}`;
    typeBadge.textContent = sample.type;
    
    // Create metrics section
    const metrics = document.createElement('div');
    metrics.className = 'metrics';
    
    // Add volume metric
    const volumeMetric = document.createElement('span');
    volumeMetric.className = 'metric';
    volumeMetric.innerHTML = `<span>Volume:</span> <strong>${sample.metrics.volume_level}%</strong>`;
    
    // Add SNR metric
    const snrMetric = document.createElement('span');
    snrMetric.className = 'metric';
    snrMetric.innerHTML = `<span>SNR:</span> <strong>${sample.metrics.signal_to_noise}dB</strong>`;
    
    // Add clarity metric
    const clarityMetric = document.createElement('span');
    clarityMetric.className = 'metric';
    clarityMetric.innerHTML = `<span>Clarity:</span> <strong>${sample.metrics.clarity}%</strong>`;
    
    // Add metrics to container
    metrics.appendChild(volumeMetric);
    metrics.appendChild(snrMetric);
    metrics.appendChild(clarityMetric);
    
    // Create actions section
    const actions = document.createElement('div');
    actions.className = 'actions';
    
    // Add play button
    const playBtn = document.createElement('button');
    playBtn.className = 'btn btn-primary btn-sm';
    playBtn.innerHTML = '<i class="fas fa-play"></i>';
    playBtn.onclick = () => playSample(sample.filename);
    
    // Add delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'btn btn-danger btn-sm';
    deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
    deleteBtn.onclick = () => deleteSample(sample.filename);
    
    // Add buttons to actions
    actions.appendChild(playBtn);
    actions.appendChild(deleteBtn);
    
    // Add all elements to row
    row.appendChild(typeBadge);
    row.appendChild(metrics);
    row.appendChild(actions);
    
    return row;
}

function updateSamplesList(samples) {
    const samplesList = document.getElementById('samplesList');
    samplesList.innerHTML = '';
    
    if (samples.length === 0) {
        samplesList.innerHTML = '<div class="no-samples">No samples recorded yet</div>';
        return;
    }
    
    samples.forEach(sample => {
        samplesList.appendChild(renderSample(sample));
    });
}

// ... rest of the existing code ... 