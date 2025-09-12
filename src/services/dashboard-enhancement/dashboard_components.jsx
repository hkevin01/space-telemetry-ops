/*
 * React Dashboard Components for Mission Control
 *
 * This module provides React components for the enhanced mission control dashboard
 * with real-time telemetry visualization, interactive widgets, and drag-and-drop
 * layout management.
 *
 * Components:
 * - DashboardGrid: Main grid container with drag-and-drop
 * - TelemetryWidget: Base widget component with chart rendering
 * - WidgetLibrary: Component library for different chart types
 * - LayoutManager: Dashboard layout configuration interface
 * - AlertPanel: Real-time alert display and management
 */

import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
    ArcElement,
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    LineElement,
    PointElement,
    Title, Tooltip
} from 'chart.js';
import {
    AlertTriangle,
    Edit,
    Minus,
    Plus,
    Settings,
    Trash2,
    TrendingDown,
    TrendingUp,
    Wifi,
    WifiOff
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { Responsive, WidthProvider } from 'react-grid-layout';

// Register Chart.js components
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Title, Tooltip, Legend
);

// Make ResponsiveGridLayout
const ResponsiveGridLayout = WidthProvider(Responsive);

// WebSocket hook for real-time data
const useWebSocket = (url, onMessage, enabled = true) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimeoutRef = useRef();

  const connect = useCallback(() => {
    if (!enabled || socket?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected');
        clearTimeout(reconnectTimeoutRef.current);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setSocket(null);
        console.log('WebSocket disconnected');

        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setSocket(ws);
    } catch (error) {
      console.error('Error creating WebSocket:', error);
    }
  }, [url, onMessage, enabled]);

  useEffect(() => {
    connect();

    return () => {
      clearTimeout(reconnectTimeoutRef.current);
      if (socket) {
        socket.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }, [socket]);

  return { socket, isConnected, sendMessage };
};

// Chart color palettes
const colorSchemes = {
  blue: { primary: '#3b82f6', secondary: '#60a5fa', background: 'rgba(59, 130, 246, 0.1)' },
  green: { primary: '#10b981', secondary: '#34d399', background: 'rgba(16, 185, 129, 0.1)' },
  red: { primary: '#ef4444', secondary: '#f87171', background: 'rgba(239, 68, 68, 0.1)' },
  yellow: { primary: '#f59e0b', secondary: '#fbbf24', background: 'rgba(245, 158, 11, 0.1)' },
  purple: { primary: '#8b5cf6', secondary: '#a78bfa', background: 'rgba(139, 92, 246, 0.1)' },
  orange: { primary: '#f97316', secondary: '#fb923c', background: 'rgba(249, 115, 22, 0.1)' },
  indigo: { primary: '#6366f1', secondary: '#818cf8', background: 'rgba(99, 102, 241, 0.1)' }
};

// Widget status indicator component
const StatusIndicator = ({ status, value, message }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'critical': return 'bg-red-500';
      case 'warning': return 'bg-yellow-500';
      case 'normal': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'critical': return <AlertTriangle className="h-6 w-6 text-white" />;
      case 'warning': return <AlertTriangle className="h-6 w-6 text-white" />;
      case 'normal': return <div className="h-6 w-6 bg-white rounded-full" />;
      default: return <Minus className="h-6 w-6 text-white" />;
    }
  };

  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className={`w-20 h-20 rounded-full ${getStatusColor(status)} flex items-center justify-center mb-4 mx-auto`}>
          {getStatusIcon(status)}
        </div>
        <div className="text-2xl font-bold">{value?.toFixed(2)}</div>
        <div className="text-sm text-gray-500 mt-1">{message}</div>
      </div>
    </div>
  );
};

// Gauge chart component
const GaugeChart = ({ value, min, max, thresholds, color, title }) => {
  const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  const getGaugeColor = () => {
    if (thresholds?.critical && value >= thresholds.critical) return '#ef4444';
    if (thresholds?.warning && value >= thresholds.warning) return '#f59e0b';
    return colorSchemes[color]?.primary || '#3b82f6';
  };

  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="relative w-32 h-32">
        <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 36 36">
          <path
            className="text-gray-300"
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="75, 25"
          />
          <path
            className="transition-all duration-500 ease-out"
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke={getGaugeColor()}
            strokeWidth="2"
            strokeDasharray={`${(percentage / 100) * 75}, 100`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-xl font-bold">{value?.toFixed(1)}</div>
            <div className="text-xs text-gray-500">{((value - min) / (max - min) * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>
      <div className="text-sm text-gray-600 mt-2 text-center">{title}</div>
    </div>
  );
};

// Line chart component with real-time updates
const LineChart = ({ data, options, thresholds }) => {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: false,
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm'
          }
        }
      },
      y: {
        beginAtZero: false,
      },
    },
    animation: {
      duration: 750,
    },
    ...options
  };

  // Add threshold lines if configured
  if (thresholds) {
    chartOptions.plugins.annotation = {
      annotations: {}
    };

    if (thresholds.warning) {
      chartOptions.plugins.annotation.annotations.warningLine = {
        type: 'line',
        yMin: thresholds.warning,
        yMax: thresholds.warning,
        borderColor: '#f59e0b',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'Warning',
          enabled: true,
          position: 'end'
        }
      };
    }

    if (thresholds.critical) {
      chartOptions.plugins.annotation.annotations.criticalLine = {
        type: 'line',
        yMin: thresholds.critical,
        yMax: thresholds.critical,
        borderColor: '#ef4444',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'Critical',
          enabled: true,
          position: 'end'
        }
      };
    }
  }

  return (
    <div className="h-full w-full">
      <Line data={data} options={chartOptions} />
    </div>
  );
};

// Individual telemetry widget component
const TelemetryWidget = ({
  widget,
  data,
  onRemove,
  onEdit,
  isEditing = false
}) => {
  const [localData, setLocalData] = useState(data);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setLocalData(data);
  }, [data]);

  const renderChart = () => {
    if (isLoading || !localData?.data) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      );
    }

    const { chart_type, data: chartData, statistics } = localData;

    switch (chart_type) {
      case 'line':
      case 'scatter':
        return (
          <LineChart
            data={chartData}
            thresholds={chartData.thresholds}
          />
        );

      case 'gauge':
        return (
          <GaugeChart
            value={chartData.value}
            min={chartData.min}
            max={chartData.max}
            thresholds={chartData.thresholds}
            color={widget.config?.color_scheme || 'blue'}
            title={widget.title}
          />
        );

      case 'status':
        return (
          <StatusIndicator
            status={chartData.status}
            value={chartData.value}
            message={chartData.message}
          />
        );

      case 'bar':
        return (
          <div className="h-full w-full">
            <Bar data={chartData} options={{ responsive: true, maintainAspectRatio: false }} />
          </div>
        );

      default:
        return (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <div className="text-lg">No Data</div>
              <div className="text-sm">Chart type: {chart_type}</div>
            </div>
          </div>
        );
    }
  };

  const getTrendIcon = () => {
    if (!localData?.statistics?.trend) return null;

    switch (localData.statistics.trend) {
      case 'increasing':
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'decreasing':
        return <TrendingDown className="h-4 w-4 text-red-500" />;
      default:
        return <Minus className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = () => {
    const { current_value } = localData?.statistics || {};
    const { warning_threshold, critical_threshold } = widget.config || {};

    if (critical_threshold !== null && current_value >= critical_threshold) {
      return <Badge variant="destructive">Critical</Badge>;
    }
    if (warning_threshold !== null && current_value >= warning_threshold) {
      return <Badge variant="secondary">Warning</Badge>;
    }
    return <Badge variant="default">Normal</Badge>;
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">{widget.title}</CardTitle>
          <div className="flex items-center space-x-2">
            {getTrendIcon()}
            {getStatusBadge()}
            {isEditing && (
              <div className="flex space-x-1">
                <Button size="sm" variant="ghost" onClick={() => onEdit?.(widget)}>
                  <Edit className="h-3 w-3" />
                </Button>
                <Button size="sm" variant="ghost" onClick={() => onRemove?.(widget.widget_id)}>
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-4 text-xs text-gray-500">
          <span>
            Current: {localData?.statistics?.current_value?.toFixed(2) || 'N/A'}
          </span>
          {localData?.statistics?.last_updated && (
            <span>
              Updated: {new Date(localData.statistics.last_updated).toLocaleTimeString()}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-2 h-[calc(100%-80px)]">
        {renderChart()}
      </CardContent>
    </Card>
  );
};

// Widget configuration dialog
const WidgetConfigDialog = ({ widget, isOpen, onClose, onSave }) => {
  const [config, setConfig] = useState({
    title: '',
    chart_type: 'line',
    data_source: '',
    spacecraft_id: '',
    time_window_hours: 24,
    color_scheme: 'blue',
    warning_threshold: '',
    critical_threshold: ''
  });

  useEffect(() => {
    if (widget) {
      setConfig({
        title: widget.title || '',
        chart_type: widget.chart_type || 'line',
        data_source: widget.data_source || '',
        spacecraft_id: widget.spacecraft_id || '',
        time_window_hours: widget.config?.time_window_hours || 24,
        color_scheme: widget.config?.color_scheme || 'blue',
        warning_threshold: widget.config?.warning_threshold?.toString() || '',
        critical_threshold: widget.config?.critical_threshold?.toString() || ''
      });
    }
  }, [widget]);

  const handleSave = () => {
    onSave({
      ...widget,
      ...config,
      warning_threshold: config.warning_threshold ? parseFloat(config.warning_threshold) : null,
      critical_threshold: config.critical_threshold ? parseFloat(config.critical_threshold) : null
    });
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{widget ? 'Edit Widget' : 'Add Widget'}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium">Title</label>
            <Input
              value={config.title}
              onChange={(e) => setConfig({...config, title: e.target.value})}
              placeholder="Widget title"
            />
          </div>

          <div>
            <label className="text-sm font-medium">Chart Type</label>
            <Select value={config.chart_type} onValueChange={(value) => setConfig({...config, chart_type: value})}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="line">Line Chart</SelectItem>
                <SelectItem value="gauge">Gauge</SelectItem>
                <SelectItem value="status">Status Indicator</SelectItem>
                <SelectItem value="bar">Bar Chart</SelectItem>
                <SelectItem value="scatter">Scatter Plot</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium">Data Source</label>
            <Input
              value={config.data_source}
              onChange={(e) => setConfig({...config, data_source: e.target.value})}
              placeholder="Telemetry parameter name"
            />
          </div>

          <div>
            <label className="text-sm font-medium">Color Scheme</label>
            <Select value={config.color_scheme} onValueChange={(value) => setConfig({...config, color_scheme: value})}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(colorSchemes).map(scheme => (
                  <SelectItem key={scheme} value={scheme}>
                    <div className="flex items-center">
                      <div
                        className="w-4 h-4 rounded mr-2"
                        style={{ backgroundColor: colorSchemes[scheme].primary }}
                      />
                      {scheme.charAt(0).toUpperCase() + scheme.slice(1)}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Warning Threshold</label>
              <Input
                type="number"
                value={config.warning_threshold}
                onChange={(e) => setConfig({...config, warning_threshold: e.target.value})}
                placeholder="Optional"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Critical Threshold</label>
              <Input
                type="number"
                value={config.critical_threshold}
                onChange={(e) => setConfig({...config, critical_threshold: e.target.value})}
                placeholder="Optional"
              />
            </div>
          </div>

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={handleSave}>Save</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// Main dashboard grid component
const DashboardGrid = ({
  layoutId,
  widgets = [],
  widgetData = {},
  isEditing = false,
  onLayoutChange,
  onWidgetRemove,
  onWidgetEdit,
  onWidgetAdd
}) => {
  const [layout, setLayout] = useState([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [editingWidget, setEditingWidget] = useState(null);

  // Convert widgets to grid layout format
  useEffect(() => {
    const gridLayout = widgets.map(widget => ({
      i: widget.widget_id,
      x: widget.position?.x || 0,
      y: widget.position?.y || 0,
      w: widget.size?.width || 4,
      h: widget.size?.height || 3,
      minW: 2,
      minH: 2
    }));
    setLayout(gridLayout);
  }, [widgets]);

  const handleLayoutChange = (newLayout) => {
    setLayout(newLayout);
    if (onLayoutChange) {
      const updatedWidgets = widgets.map(widget => {
        const layoutItem = newLayout.find(item => item.i === widget.widget_id);
        if (layoutItem) {
          return {
            ...widget,
            position: { x: layoutItem.x, y: layoutItem.y },
            size: { width: layoutItem.w, height: layoutItem.h }
          };
        }
        return widget;
      });
      onLayoutChange(updatedWidgets);
    }
  };

  const handleWidgetEdit = (widget) => {
    setEditingWidget(widget);
  };

  const handleWidgetSave = (updatedWidget) => {
    if (onWidgetEdit) {
      onWidgetEdit(updatedWidget);
    }
    setEditingWidget(null);
  };

  const handleWidgetAdd = (newWidget) => {
    if (onWidgetAdd) {
      onWidgetAdd(newWidget);
    }
    setShowAddDialog(false);
  };

  return (
    <div className="h-full">
      {isEditing && (
        <div className="mb-4 flex justify-end">
          <Button onClick={() => setShowAddDialog(true)} className="flex items-center">
            <Plus className="h-4 w-4 mr-2" />
            Add Widget
          </Button>
        </div>
      )}

      <ResponsiveGridLayout
        className="layout"
        layouts={{ lg: layout }}
        breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
        rowHeight={60}
        onLayoutChange={handleLayoutChange}
        isDraggable={isEditing}
        isResizable={isEditing}
        margin={[16, 16]}
      >
        {widgets.map(widget => (
          <div key={widget.widget_id}>
            <TelemetryWidget
              widget={widget}
              data={widgetData[widget.widget_id]}
              onRemove={onWidgetRemove}
              onEdit={handleWidgetEdit}
              isEditing={isEditing}
            />
          </div>
        ))}
      </ResponsiveGridLayout>

      <WidgetConfigDialog
        widget={editingWidget}
        isOpen={!!editingWidget}
        onClose={() => setEditingWidget(null)}
        onSave={handleWidgetSave}
      />

      <WidgetConfigDialog
        widget={null}
        isOpen={showAddDialog}
        onClose={() => setShowAddDialog(false)}
        onSave={handleWidgetAdd}
      />
    </div>
  );
};

// Alert panel component
const AlertPanel = ({ alerts = [], onAlertAcknowledge, onAlertDismiss }) => {
  const [expandedAlert, setExpandedAlert] = useState(null);

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-yellow-500 bg-yellow-50';
      case 'info': return 'border-blue-500 bg-blue-50';
      default: return 'border-gray-500 bg-gray-50';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
      case 'warning':
        return <AlertTriangle className="h-5 w-5" />;
      default:
        return <AlertTriangle className="h-5 w-5" />;
    }
  };

  if (alerts.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <div className="text-lg font-medium">All Clear</div>
        <div className="text-sm">No active alerts</div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {alerts.map(alert => (
        <Alert key={alert.id} className={getSeverityColor(alert.severity)}>
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              {getSeverityIcon(alert.severity)}
              <div className="flex-1">
                <AlertDescription className="font-medium">
                  {alert.message}
                </AlertDescription>
                {expandedAlert === alert.id && (
                  <div className="mt-2 text-sm space-y-1">
                    <div><strong>Source:</strong> {alert.source}</div>
                    <div><strong>Time:</strong> {new Date(alert.timestamp).toLocaleString()}</div>
                    {alert.details && (
                      <div><strong>Details:</strong> {alert.details}</div>
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setExpandedAlert(
                  expandedAlert === alert.id ? null : alert.id
                )}
              >
                {expandedAlert === alert.id ? 'Less' : 'More'}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onAlertAcknowledge?.(alert.id)}
              >
                Acknowledge
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onAlertDismiss?.(alert.id)}
              >
                Dismiss
              </Button>
            </div>
          </div>
        </Alert>
      ))}
    </div>
  );
};

// Main mission control dashboard component
const MissionControlDashboard = ({ layoutId, apiEndpoint = '/api/dashboard' }) => {
  const [layout, setLayout] = useState(null);
  const [widgets, setWidgets] = useState([]);
  const [widgetData, setWidgetData] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // WebSocket connection for real-time updates
  const { isConnected, sendMessage } = useWebSocket(
    `ws://localhost:8000${apiEndpoint}/ws/${layoutId}`,
    useCallback((data) => {
      switch (data.type) {
        case 'layout_data':
          setLayout(data.layout);
          setWidgets(data.layout.widgets || []);
          break;
        case 'widget_update':
          setWidgetData(prev => ({
            ...prev,
            [data.widget_id]: data.data
          }));
          break;
        case 'system_alert':
          setAlerts(prev => [data.alert, ...prev.slice(0, 49)]); // Keep last 50 alerts
          break;
        default:
          console.log('Unknown WebSocket message type:', data.type);
      }
    }, [])
  );

  // Load initial layout data
  useEffect(() => {
    const fetchLayout = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`${apiEndpoint}/layouts/${layoutId}`);
        if (!response.ok) throw new Error('Failed to fetch layout');

        const layoutData = await response.json();
        setLayout(layoutData);
        setWidgets(layoutData.widgets || []);

        // Fetch initial widget data
        const widgetDataPromises = (layoutData.widgets || []).map(async widget => {
          const dataResponse = await fetch(
            `${apiEndpoint}/layouts/${layoutId}/widgets/${widget.widget_id}/data`
          );
          if (dataResponse.ok) {
            const data = await dataResponse.json();
            return [widget.widget_id, data];
          }
          return [widget.widget_id, null];
        });

        const widgetDataResults = await Promise.all(widgetDataPromises);
        const initialWidgetData = Object.fromEntries(widgetDataResults);
        setWidgetData(initialWidgetData);

      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    if (layoutId) {
      fetchLayout();
    }
  }, [layoutId, apiEndpoint]);

  const handleLayoutChange = async (updatedWidgets) => {
    // Update layout on server
    try {
      const response = await fetch(`${apiEndpoint}/layouts/${layoutId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: layout.name,
          description: layout.description,
          mission_id: layout.mission_id,
          theme: layout.config?.theme || 'dark'
        })
      });

      if (!response.ok) throw new Error('Failed to update layout');

      setWidgets(updatedWidgets);
    } catch (err) {
      console.error('Error updating layout:', err);
    }
  };

  const handleWidgetRemove = async (widgetId) => {
    try {
      const response = await fetch(
        `${apiEndpoint}/layouts/${layoutId}/widgets/${widgetId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) throw new Error('Failed to remove widget');

      setWidgets(prev => prev.filter(w => w.widget_id !== widgetId));
      setWidgetData(prev => {
        const newData = { ...prev };
        delete newData[widgetId];
        return newData;
      });
    } catch (err) {
      console.error('Error removing widget:', err);
    }
  };

  const handleWidgetEdit = async (updatedWidget) => {
    // Update widget on server
    try {
      // For now, we'll just update locally
      setWidgets(prev => prev.map(w =>
        w.widget_id === updatedWidget.widget_id ? updatedWidget : w
      ));
    } catch (err) {
      console.error('Error updating widget:', err);
    }
  };

  const handleWidgetAdd = async (newWidget) => {
    try {
      const response = await fetch(`${apiEndpoint}/layouts/${layoutId}/widgets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newWidget)
      });

      if (!response.ok) throw new Error('Failed to add widget');

      const result = await response.json();
      setWidgets(prev => [...prev, result.widget]);
    } catch (err) {
      console.error('Error adding widget:', err);
    }
  };

  const handleAlertAcknowledge = (alertId) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  const handleAlertDismiss = (alertId) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-lg">Loading dashboard...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert className="max-w-md">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Error loading dashboard: {error}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold">{layout?.name || 'Mission Control'}</h1>
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <Badge variant="default" className="flex items-center">
                <Wifi className="h-3 w-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge variant="destructive" className="flex items-center">
                <WifiOff className="h-3 w-3 mr-1" />
                Disconnected
              </Badge>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={isEditing ? "default" : "outline"}
            onClick={() => setIsEditing(!isEditing)}
          >
            <Settings className="h-4 w-4 mr-2" />
            {isEditing ? 'Done' : 'Edit'}
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Dashboard grid */}
        <div className="flex-1 p-4">
          <DashboardGrid
            layoutId={layoutId}
            widgets={widgets}
            widgetData={widgetData}
            isEditing={isEditing}
            onLayoutChange={handleLayoutChange}
            onWidgetRemove={handleWidgetRemove}
            onWidgetEdit={handleWidgetEdit}
            onWidgetAdd={handleWidgetAdd}
          />
        </div>

        {/* Alert panel */}
        {alerts.length > 0 && (
          <div className="w-96 border-l p-4">
            <h2 className="text-lg font-semibold mb-4">
              Active Alerts ({alerts.length})
            </h2>
            <AlertPanel
              alerts={alerts}
              onAlertAcknowledge={handleAlertAcknowledge}
              onAlertDismiss={handleAlertDismiss}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default MissionControlDashboard;
export {
    AlertPanel, DashboardGrid, GaugeChart,
    LineChart, StatusIndicator, TelemetryWidget, WidgetConfigDialog
};

