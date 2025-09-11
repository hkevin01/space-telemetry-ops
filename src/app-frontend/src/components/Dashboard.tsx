import {
    Activity,
    AlertTriangle,
    Battery,
    CheckCircle,
    Clock,
    Minus,
    Satellite,
    Signal,
    Thermometer,
    TrendingDown,
    TrendingUp
} from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';

// Types and interfaces
interface SpacecraftData {
  id: string;
  name: string;
  mission_name: string;
  spacecraft_id: string;
  mission_phase: string;
  is_active: boolean;
  battery_level?: number;
  temperature?: number;
  signal_strength?: number;
  overall_health_score?: number;
  communication_status: string;
  last_contact?: string;
}

interface TelemetryPacket {
  id: string;
  spacecraft_time: string;
  ground_received_time: string;
  packet_id: string;
  sequence_number: number;
  telemetry_status: string;
  data_quality: string;
  spacecraft_id: string;
  alert_level: number;
  processed_data: Record<string, any>;
}

interface HealthMetrics {
  status: string;
  timestamp: string;
  database_connected: boolean;
  memory_usage_mb: number;
  processing_queue_size: number;
  last_telemetry_received?: string;
}

// Utility functions
const formatDateTime = (dateString: string) => {
  return new Date(dateString).toLocaleString();
};

const formatTimeAgo = (dateString: string) => {
  const now = new Date();
  const then = new Date(dateString);
  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
};

const getStatusColor = (status: string) => {
  switch (status.toLowerCase()) {
    case 'nominal':
    case 'healthy':
      return 'text-green-600 bg-green-100';
    case 'warning':
      return 'text-yellow-600 bg-yellow-100';
    case 'critical':
    case 'error':
      return 'text-red-600 bg-red-100';
    default:
      return 'text-gray-600 bg-gray-100';
  }
};

const getHealthIcon = (score?: number) => {
  if (!score) return <Minus className="w-4 h-4" />;

  if (score >= 0.8) return <CheckCircle className="w-4 h-4 text-green-600" />;
  if (score >= 0.6) return <AlertTriangle className="w-4 h-4 text-yellow-600" />;
  return <AlertTriangle className="w-4 h-4 text-red-600" />;
};

const getTrendIcon = (current: number, previous: number) => {
  if (current > previous) return <TrendingUp className="w-4 h-4 text-green-600" />;
  if (current < previous) return <TrendingDown className="w-4 h-4 text-red-600" />;
  return <Minus className="w-4 h-4 text-gray-400" />;
};

// Main Dashboard Component
const SpaceTelemetryDashboard: React.FC = () => {
  const [spacecraft, setSpacecraft] = useState<SpacecraftData[]>([]);
  const [telemetryPackets, setTelemetryPackets] = useState<TelemetryPacket[]>([]);
  const [healthMetrics, setHealthMetrics] = useState<HealthMetrics | null>(null);
  const [selectedSpacecraft, setSelectedSpacecraft] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // API base URL - in production, this would come from environment variables
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

  // Fetch functions
  const fetchSpacecraft = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/spacecraft/`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setSpacecraft(data);
    } catch (err) {
      console.error('Failed to fetch spacecraft:', err);
      setError('Failed to load spacecraft data');
    }
  }, [API_BASE]);

  const fetchTelemetry = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (selectedSpacecraft) {
        params.append('spacecraft_id', selectedSpacecraft);
      }
      params.append('limit', '50');

      const response = await fetch(`${API_BASE}/telemetry/?${params}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setTelemetryPackets(data);
    } catch (err) {
      console.error('Failed to fetch telemetry:', err);
      setError('Failed to load telemetry data');
    }
  }, [API_BASE, selectedSpacecraft]);

  const fetchHealthMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health/`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setHealthMetrics(data);
    } catch (err) {
      console.error('Failed to fetch health metrics:', err);
    }
  }, [API_BASE]);

  // Initial load and auto-refresh
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchSpacecraft(),
        fetchTelemetry(),
        fetchHealthMetrics()
      ]);
      setLoading(false);
    };

    loadData();
  }, [fetchSpacecraft, fetchTelemetry, fetchHealthMetrics]);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchTelemetry();
      fetchHealthMetrics();
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, fetchTelemetry, fetchHealthMetrics]);

  // Handle spacecraft selection
  const handleSpacecraftSelect = useCallback((spacecraftId: string) => {
    setSelectedSpacecraft(spacecraftId === selectedSpacecraft ? null : spacecraftId);
  }, [selectedSpacecraft]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-700">Loading Telemetry Data...</h2>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 text-red-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Satellite className="w-8 h-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">
                Space Telemetry Operations
              </h1>
            </div>

            <div className="flex items-center space-x-4">
              {/* System Health Indicator */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  healthMetrics?.status === 'healthy' ? 'bg-green-500' :
                  healthMetrics?.status === 'warning' ? 'bg-yellow-500' :
                  'bg-red-500'
                }`} />
                <span className="text-sm text-gray-600">
                  System {healthMetrics?.status || 'Unknown'}
                </span>
              </div>

              {/* Auto-refresh toggle */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm text-gray-600">Auto-refresh</span>
              </label>

              {/* Last update time */}
              {healthMetrics && (
                <span className="text-sm text-gray-500">
                  Updated {formatTimeAgo(healthMetrics.timestamp)}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Active Spacecraft"
            value={spacecraft.filter(sc => sc.is_active).length.toString()}
            icon={<Satellite className="w-6 h-6" />}
            color="blue"
          />

          <MetricCard
            title="Memory Usage"
            value={`${healthMetrics?.memory_usage_mb?.toFixed(0) || 0} MB`}
            icon={<Activity className="w-6 h-6" />}
            color="purple"
          />

          <MetricCard
            title="Queue Size"
            value={healthMetrics?.processing_queue_size?.toString() || '0'}
            icon={<Clock className="w-6 h-6" />}
            color="orange"
          />

          <MetricCard
            title="DB Status"
            value={healthMetrics?.database_connected ? 'Connected' : 'Disconnected'}
            icon={<CheckCircle className="w-6 h-6" />}
            color={healthMetrics?.database_connected ? 'green' : 'red'}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Spacecraft List */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-gray-900">Spacecraft</h3>
              </div>
              <div className="divide-y">
                {spacecraft.map((sc) => (
                  <SpacecraftCard
                    key={sc.id}
                    spacecraft={sc}
                    selected={selectedSpacecraft === sc.id}
                    onSelect={() => handleSpacecraftSelect(sc.id)}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Telemetry Feed */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-gray-900">
                  Recent Telemetry
                  {selectedSpacecraft && (
                    <span className="text-sm text-gray-500 ml-2">
                      ({spacecraft.find(sc => sc.id === selectedSpacecraft)?.name})
                    </span>
                  )}
                </h3>
              </div>
              <div className="max-h-96 overflow-y-auto">
                {telemetryPackets.length === 0 ? (
                  <div className="p-6 text-center text-gray-500">
                    No telemetry data available
                  </div>
                ) : (
                  telemetryPackets.map((packet) => (
                    <TelemetryPacketCard
                      key={packet.id}
                      packet={packet}
                      spacecraft={spacecraft.find(sc => sc.id === packet.spacecraft_id)}
                    />
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Metric Card Component
interface MetricCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon, color }) => {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    purple: 'text-purple-600 bg-purple-100',
    orange: 'text-orange-600 bg-orange-100',
    green: 'text-green-600 bg-green-100',
    red: 'text-red-600 bg-red-100'
  }[color] || 'text-gray-600 bg-gray-100';

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center">
        <div className={`p-2 rounded-md ${colorClasses}`}>
          {icon}
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
};

// Spacecraft Card Component
interface SpacecraftCardProps {
  spacecraft: SpacecraftData;
  selected: boolean;
  onSelect: () => void;
}

const SpacecraftCard: React.FC<SpacecraftCardProps> = ({ spacecraft, selected, onSelect }) => {
  return (
    <div
      className={`p-4 cursor-pointer transition-colors ${
        selected ? 'bg-blue-50 border-r-4 border-blue-500' : 'hover:bg-gray-50'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h4 className="font-medium text-gray-900">{spacecraft.name}</h4>
          <p className="text-sm text-gray-600">{spacecraft.mission_name}</p>
          <div className="flex items-center mt-2 space-x-4">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              getStatusColor(spacecraft.communication_status)
            }`}>
              {spacecraft.communication_status}
            </span>
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              spacecraft.is_active ? 'text-green-600 bg-green-100' : 'text-gray-600 bg-gray-100'
            }`}>
              {spacecraft.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>

        <div className="flex flex-col items-end space-y-1">
          {getHealthIcon(spacecraft.overall_health_score)}

          <div className="flex items-center space-x-2 text-xs text-gray-500">
            {spacecraft.battery_level !== undefined && (
              <div className="flex items-center">
                <Battery className="w-3 h-3 mr-1" />
                {spacecraft.battery_level}%
              </div>
            )}

            {spacecraft.temperature !== undefined && (
              <div className="flex items-center">
                <Thermometer className="w-3 h-3 mr-1" />
                {spacecraft.temperature}°C
              </div>
            )}

            {spacecraft.signal_strength !== undefined && (
              <div className="flex items-center">
                <Signal className="w-3 h-3 mr-1" />
                {spacecraft.signal_strength}dBm
              </div>
            )}
          </div>

          {spacecraft.last_contact && (
            <span className="text-xs text-gray-500">
              {formatTimeAgo(spacecraft.last_contact)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

// Telemetry Packet Card Component
interface TelemetryPacketCardProps {
  packet: TelemetryPacket;
  spacecraft?: SpacecraftData;
}

const TelemetryPacketCard: React.FC<TelemetryPacketCardProps> = ({ packet, spacecraft }) => {
  return (
    <div className="p-4 border-b border-gray-100 hover:bg-gray-50">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <h4 className="font-medium text-gray-900">
              {packet.packet_id} #{packet.sequence_number}
            </h4>
            {spacecraft && (
              <span className="text-sm text-gray-600">
                from {spacecraft.name}
              </span>
            )}
          </div>

          <div className="flex items-center mt-1 space-x-4">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              getStatusColor(packet.telemetry_status)
            }`}>
              {packet.telemetry_status}
            </span>

            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              getStatusColor(packet.data_quality)
            }`}>
              {packet.data_quality} quality
            </span>

            {packet.alert_level > 0 && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-red-600 bg-red-100">
                Alert Level {packet.alert_level}
              </span>
            )}
          </div>

          {/* Data preview */}
          <div className="mt-2 text-xs text-gray-600">
            {Object.keys(packet.processed_data).slice(0, 3).map(key => (
              <span key={key} className="mr-4">
                {key}: {JSON.stringify(packet.processed_data[key])}
              </span>
            ))}
          </div>
        </div>

        <div className="text-right">
          <div className="text-sm text-gray-900">
            {formatTimeAgo(packet.ground_received_time)}
          </div>
          <div className="text-xs text-gray-500">
            SC: {formatDateTime(packet.spacecraft_time)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpaceTelemetryDashboard;
