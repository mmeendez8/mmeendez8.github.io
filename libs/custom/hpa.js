document.addEventListener('DOMContentLoaded', function () {
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
            },
            plugins: {
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: {
                            enabled: true,
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy',
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
    switch (x_axis) {
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

    let xMax = x_axis_metric * 5;

    // Generate values for plotting
    for (let x = start; x <= xMax; x += x_axis_metric / 1000) {
        xValues.push(x);
        let replicas = calculateReplicas(x);
        yValues.push(replicas);
    }


    let yMax = Math.max(...yValues.filter(val => val !== Infinity));
    let yMin = 0
    let xMin = Math.min(x_axis_metric - 1 / 2 * x_axis_metric, start)

    chart.data.labels = xValues;
    chart.data.datasets[0].data = yValues;
    chart.data.datasets[1].data = [{ x: x_axis_metric, y: yMin }, { x: x_axis_metric, y: yMax }],

        chart.resetZoom();

    chart.options.scales.x.min = x_axis_metric - 1 / 2 * x_axis_metric;
    chart.options.scales.x.max = x_axis_metric + 1 / 2 * x_axis_metric;
    chart.options.scales.y.max = calculateReplicas(x_axis_metric) + 2;


    chart.update();
}


