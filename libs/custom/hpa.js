document.addEventListener('DOMContentLoaded', function() {
    const currentReplicasInput = document.getElementById('current_replicas');
    const currentMetricValueInput = document.getElementById('current_metric_value');
    const targetMetricValueInput = document.getElementById('target_metric_value');
    const xAxisSelect = document.getElementById('xAxisSelect');
    const ctx = document.getElementById('replicasChart').getContext('2d');
    const chart = createChart(ctx); 


    const formElements = [currentReplicasInput, currentMetricValueInput, targetMetricValueInput, xAxisSelect];

    formElements.forEach(element => {
        element.addEventListener('change', () => updateChart(chart, currentReplicasInput, currentMetricValueInput, targetMetricValueInput, xAxisSelect));
    });
    
    updateChart(chart, currentReplicasInput, currentMetricValueInput, targetMetricValueInput, xAxisSelect)
});

function createChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Desired Replicas',
                backgroundColor: 'rgba(58, 134, 255, 1)',
                borderColor: 'rgba(58, 134, 255, 1)',
                data: [],
                fill: false,
                pointRadius: 0
            },
            {
                label: 'Current Value',
                data: [],
                backgroundColor: 'rgba(251, 86, 7, 1)',
                borderColor: 'rgba(251, 86, 7, 1)',
                borderWidth: 2,
                fill: false,
                pointRadius: 0,
                showLine: true
            }
        ]
        },
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Desired Replicas'
                    }
                }
            }
        }
    });
}

function updateChart(chart, currentReplicasInput, currentMetricValueInput, targetMetricValueInput, xAxisSelect) {
    let current_replicas = parseInt(currentReplicasInput.value);
    let current_metric_value = parseFloat(currentMetricValueInput.value);
    let target_metric_value = parseFloat(targetMetricValueInput.value);
    let x_axis = xAxisSelect.value;

    let xValues = [];
    let yValues = [];

    let start = 0
    switch(x_axis) {
        case 'current_metric':
            x_axis_label = 'Current Metric Value';
            x_axis_metric = current_metric_value;
            calculateReplicas = x => Math.ceil(current_replicas * (x / target_metric_value));
            break;
        case 'target_metric':
            x_axis_label = 'Target Metric Value';
            x_axis_metric = target_metric_value;
            calculateReplicas = x => Math.ceil(current_replicas * (current_metric_value / x));
            break;
        case 'replicas':
            x_axis_label = 'Current Replicas';
            x_axis_metric = current_replicas;
            calculateReplicas = x => Math.ceil(x * (current_metric_value / target_metric_value));
            start = 1;
            break;
    }

    // Generate values for plotting
    for (let x = start; x <= Math.min(2 * x_axis_metric, 10); x += x_axis_metric / 1000) {
        xValues.push(x);
        let replicas = calculateReplicas(x);
        yValues.push(replicas);
    }
    
    let maxYValue = Math.max(...yValues.filter(val => val !== Infinity));
    chart.data.labels = xValues;
    chart.data.datasets[0].data = yValues;
    chart.data.datasets[1].data = [{x: x_axis_metric, y: 0}, {x: x_axis_metric, y: maxYValue}],
    // chart.options.scales.y.title.text = x_axis_label;
    chart.update();
}


